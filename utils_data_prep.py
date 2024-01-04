import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata



def get_common_genes(dataset_paths):
    gene_sets = []
    for path in dataset_paths:
        with h5py.File(path, 'r') as file:
            genes = list(file['exp']['rowData'])
            gene_sets.append(set(genes))
    
    common_genes = set.intersection(*gene_sets)
    return common_genes


def filter_counts_and_save(input_file_path, output_file_path, common_genes):
    if not output_file_path.exists():
        with h5py.File(input_file_path, 'r') as file:
            # Read the existing data
            all_genes = list(file['exp']['rowData'])
            counts = file['exp']['counts'][:]
            col_data = file['exp']['colData'][:] 
            # Determine the indices of the common genes
            gene_indices = [all_genes.index(gene) for gene in common_genes if gene in all_genes]

            # Filter the counts data
            filtered_counts = counts[:, gene_indices]

            # Filter the gene names
            filtered_gene_names = [all_genes[i] for i in gene_indices]
        # Write the filtered data to a new .h5 file
        with h5py.File(output_file_path, 'w') as new_file:
            # Create a new group named 'exp'
            exp_group = new_file.create_group('exp')

            # Create a new dataset for filtered rowData
            new_file.create_dataset('exp/rowData', data=filtered_gene_names)

            # Create a new dataset for filtered counts
            new_file.create_dataset('exp/counts', data=filtered_counts)
            new_file.create_dataset('exp/colData', data=col_data)


def reduce_to_common_genes(viable_conc, dataset_type, OUTDIR):
    dataset_paths = [OUTDIR / f'{dataset_type}.{conc}_mikrog.GEX_cellrangerADT_SCE.h5' for conc in viable_conc]
    common_genes = get_common_genes(dataset_paths)

    for conc in viable_conc:
        input_file_path = OUTDIR / f'{dataset_type}.{conc}_mikrog.GEX_cellrangerADT_SCE.h5'
        output_file_path = OUTDIR / f'{dataset_type}.{conc}_mikrog.GEX_cellrangerADT_SCE_common_genes.h5'
        filter_counts_and_save(input_file_path, output_file_path, common_genes)

def create_dataset(viable_conc, dataset_type, OUTDIR):
    reduce_to_common_genes(viable_conc, dataset_type, OUTDIR)
    all_counts = []
    all_counts_no_gex = []
    all_metadata = []

    for conc in viable_conc:
        path = OUTDIR / f'{dataset_type}.{conc}_mikrog.GEX_cellrangerADT_SCE_common_genes.h5'
        with h5py.File(path, 'r') as file:
            # Load datasets
            exp_group = file['exp']
            col_data = exp_group['colData']
            counts = exp_group['counts']
            #fields = ['celltype_final']
            ## Print unique entries for each field
            #for field in fields:
            #    unique_entries = np.unique(col_data[field])
            #    print(f"Unique entries from conc {conc} in '{field}': {unique_entries}")

            # Filter for fractionMT
            fractionMT_filter = col_data['fractionMT'] <= 0.2

            # Calculate 10th percentiles for n_gene and n_umi
            n_gene_10th_percentile = np.percentile(col_data['n_gene'], 10)
            n_umi_10th_percentile = np.percentile(col_data['n_umi'], 10)

            # Create filters for n_gene and n_umi
            n_gene_filter = col_data['n_gene'] > n_gene_10th_percentile
            n_umi_filter = col_data['n_umi'] > n_umi_10th_percentile

            # Combine all filters
            combined_filter = fractionMT_filter & n_gene_filter & n_umi_filter

            # Convert boolean mask to indices
            indices = np.where(combined_filter)[0]

            # Apply filter to counts dataset using the indices
            filtered_counts = counts[indices, :]

            # Apply the same filter to colData dataset
            filtered_col_data = col_data[indices]

            #print(f"Number of cells from conc {conc} after filtering:", filtered_counts.shape[0])

            # Identify ADT_ fields excluding ADT_Hashtag fields
            adt_fields = [field for field in col_data.dtype.names if field.startswith('ADT_') and 'Hashtag' not in field and 'barcodes' not in field]
            print("entry of first ADT field:", col_data[adt_fields[0]][0])
            # Extract these fields for the filtered cells
            adt_data = np.array([filtered_col_data[field] for field in adt_fields]).T
            no_ADT_fields = adt_data.shape[1]
            # Concatenate these fields with the filtered count dataset
            # Assuming filtered_counts and adt_data are numpy arrays
            combined_data = np.hstack((adt_data, filtered_counts))
            n_cells = filtered_counts.shape[0]
            conc_data = np.full(n_cells, conc)  # Create an array filled with the current concentration

            # Create a DataFrame for the metadata
            metadata_df = pd.DataFrame({
                'cycle_phase': filtered_col_data['cycle_phase'],
                'celltype_final': filtered_col_data['celltype_final'],
                'conc': conc_data
            })

            # Append the combined data and metadata to their respective lists
            all_counts.append(combined_data)
            all_counts_no_gex.append(adt_data)
            all_metadata.append(metadata_df)

    # Concatenate all counts and metadata
    concatenated_counts = np.vstack(all_counts)
    concatenated_counts_no_gex = np.vstack(all_counts_no_gex)
    concatenated_metadata = pd.concat(all_metadata, ignore_index=True)

    return concatenated_counts, concatenated_counts_no_gex, concatenated_metadata, no_ADT_fields


def only_highly_variable_genes(adata, non_gene_features):
    cell_counts = adata.obs['celltype_final'].value_counts()
    cell_types_to_keep = cell_counts[cell_counts >= 10].index

    # Filter the AnnData object to only include these cell types
    adata_filtered = adata[adata.obs['celltype_final'].isin(cell_types_to_keep)].copy()
    adata = adata_filtered


    # Get unique cell types
    cell_types = adata.obs['celltype_final'].unique()


    # Initialize a DataFrame to store within-cell-type variances
    within_variability = pd.DataFrame(index=adata.var_names)

    for cell_type in cell_types:
        subset = adata[adata.obs['celltype_final'] == cell_type]
        within_variability[cell_type] = subset.X.var(axis=0)

    #print("head of within_variability: ",within_variability.head())
    #print("end of within_variability: ",within_variability.tail())
    #print("shape of within_variability: ",within_variability.shape)
    # Calculate mean expression of each gene in each cell type
    mean_expression = adata.to_df().groupby(adata.obs['celltype_final']).mean()

    # Calculate variance across cell types
    across_variability = mean_expression.var(axis=0)


    # Rank genes by across-cell-type variability
    ranked_genes = across_variability.sort_values(ascending=False).index

    # Filter based on within-cell-type variability
    # You might need to adjust the threshold based on your data
    threshold =  0.5 * within_variability.median().median()  # Example threshold
    filtered_genes = [gene for gene in ranked_genes if within_variability.loc[gene].max() < threshold]

    # Select top 200 genes
    top_200_genes = filtered_genes[:200]

    non_gene_column_names = [str(i) for i in range(non_gene_features)]  # Names of non-gene columns

    # Combine non-gene column names with top gene names
    column_names_to_keep = non_gene_column_names + top_200_genes

    # Update AnnData object to keep only the columns with these names
    adata = adata[:, column_names_to_keep]
    return adata