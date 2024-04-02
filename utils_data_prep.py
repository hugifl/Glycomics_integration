import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def find_celltype_marker_genes(AB_final_metadata, lectin_final_metadata, AB_log_transformed_gene_data, lectin_log_transformed_gene_data, common_genes, OUTDIR):
    # Combine the two datasets
    combined_gene_data = np.concatenate([AB_log_transformed_gene_data, lectin_log_transformed_gene_data], axis=0)
    combined_metadata = pd.concat([AB_final_metadata, lectin_final_metadata])

    # Check if 'celltype_major' column exists in the metadata
    if 'celltype_major' not in combined_metadata.columns:
        raise ValueError("Metadata must contain a 'celltype_major' column.")

    # Extract cell type labels
    cell_types = combined_metadata['celltype_major']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(combined_gene_data, cell_types, test_size=0.3, random_state=42, stratify=cell_types)

    print("initializing and training the Random Forest classifier")
    # Initialize and train the Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    print("predicting on the test set and print the classification report")

    # Predict on the test set and print the classification report
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Calculate feature importances
    importances = rf.feature_importances_

    # Sort the feature importances in descending order and map them to gene names
    indices = np.argsort(importances)[::-1]
    sorted_gene_names = [common_genes[i] for i in indices]

    print("plotting the feature importances")
    accuracies = []
    for i in range(1, 100 + 1):
        print(f"Top {i} out of 100 features being tested")
        # Train with the top i features
        top_i_indices = indices[:i]
        rf.fit(X_train[:, top_i_indices], y_train)
        y_pred = rf.predict(X_test[:, top_i_indices])
        accuracies.append(accuracy_score(y_test, y_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, label='Model Accuracy')
    plt.title('Feature Importance and Model Performance')
    plt.xlabel('Number of Top Features Used')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTDIR / 'feature_importance_performance.png')

    # Return both indices and gene names along with their importance scores
    return [(idx, gene, importances[idx]) for idx, gene in zip(indices, sorted_gene_names)], sorted_gene_names


def filter_counts_and_save(input_file_path, output_file_path, common_genes):
    if not output_file_path.exists():
        with h5py.File(input_file_path, 'r') as file:
            # Read the existing data
            all_genes = list(file['exp']['rowData'])
            print("number of all genes: ", len(all_genes))
            counts = file['exp']['counts'][:]
            print("shape of counts: ", counts.shape)
            col_data = file['exp']['colData'][:] 

            assert all(gene in all_genes for gene in common_genes), "Some common genes are missing."
            # Determine the indices of the common genes
            gene_indices = [all_genes.index(gene) for gene in common_genes if gene in all_genes]

            # Filter the counts data
            filtered_counts = counts[:, gene_indices]
            print("shape of filtered counts: ", filtered_counts.shape)

            # Filter the gene names
            filtered_gene_names = common_genes
            #filtered_gene_names = [all_genes[i] for i in gene_indices]
            print("number of filtered genes: ", len(filtered_gene_names))
        # Write the filtered data to a new .h5 file
        with h5py.File(output_file_path, 'w') as new_file:
            # Create a new group named 'exp'
            exp_group = new_file.create_group('exp')

            # Create a new dataset for filtered rowData
            new_file.create_dataset('exp/rowData', data=filtered_gene_names)

            # Create a new dataset for filtered counts
            new_file.create_dataset('exp/counts', data=filtered_counts)
            new_file.create_dataset('exp/colData', data=col_data)

def reduce_to_common_genes(viable_conc_1, dataset_type_1, viable_conc_2, dataset_type_2, OUTDIR):
    dataset_paths_1 = [OUTDIR / f'{dataset_type_1}.{conc}_mikrog.GEX_cellrangerADT_SCE.h5' for conc in viable_conc_1]
    dataset_paths_2 = [OUTDIR / f'{dataset_type_2}.{conc}_mikrog.GEX_cellrangerADT_SCE.h5' for conc in viable_conc_2]

    print("computing common genes")
    common_genes = get_common_genes(dataset_paths_1, dataset_paths_2)

    print("filtering and saving dataset 1")
    for conc in viable_conc_1:
        print("conc: ", conc)
        input_file_path = OUTDIR / f'{dataset_type_1}.{conc}_mikrog.GEX_cellrangerADT_SCE.h5'
        output_file_path = OUTDIR / f'{dataset_type_1}.{conc}_mikrog.GEX_cellrangerADT_SCE_common_genes.h5'
        filter_counts_and_save(input_file_path, output_file_path, common_genes)

    print("filtering and saving dataset 2")
    for conc in viable_conc_2:
        print("conc: ", conc)
        input_file_path = OUTDIR / f'{dataset_type_2}.{conc}_mikrog.GEX_cellrangerADT_SCE.h5'
        output_file_path = OUTDIR / f'{dataset_type_2}.{conc}_mikrog.GEX_cellrangerADT_SCE_common_genes.h5'
        filter_counts_and_save(input_file_path, output_file_path, common_genes)

    return common_genes

def get_common_genes(dataset_paths_1, dataset_paths_2):
    gene_sets = []
    
    for path in dataset_paths_2:
        with h5py.File(path, 'r') as file:
            genes = list(file['exp']['rowData'])
            gene_sets.append(set(genes))
    
    for path in dataset_paths_1:
        with h5py.File(path, 'r') as file:
            genes = list(file['exp']['rowData'])
            gene_sets.append(set(genes))

    common_genes = set.intersection(*gene_sets)
    common_genes = sorted(list(common_genes))
    print("number of common genes: ", len(common_genes))
    return common_genes


def test_reduced_genes(viable_conc_1, dataset_type_1, viable_conc_2, dataset_type_2, OUTDIR):
    # Initialize variables to store reference dimensions and gene names
    ref_dims = None
    ref_genes = None

    # Iterate through each dataset
    for dataset_type in [dataset_type_1, dataset_type_2]:
        if dataset_type == dataset_type_1:
            viable_conc = viable_conc_1
        else:
            viable_conc = viable_conc_2
        for conc in viable_conc:
            path = OUTDIR / f'{dataset_type}.{conc}_mikrog.GEX_cellrangerADT_SCE_common_genes.h5'
            
            with h5py.File(path, 'r') as file:
                # Access the datasets
                exp_group = file['exp']
                counts = exp_group['counts']
                genes = exp_group['rowData']

                # Check dimensions
                if ref_dims is None:
                    ref_dims = counts.shape[1]  # Number of columns (genes)
                else:
                    if ref_dims != counts.shape[1]:
                        print(f"Dimension mismatch in {dataset_type} {conc}")
                        print(f"Expected {ref_dims}, got {counts.shape[1]}")

                # Check gene names and order
                if ref_genes is None:
                    ref_genes = genes[:]
                else:
                    if not np.array_equal(ref_genes, genes[:]):
                        print(f"Gene mismatch or different order in {dataset_type} {conc}")

    print("Test completed.")


def load_filter_combine_data(viable_conc, dataset_type, OUTDIR, celltypes_to_keep, percentile):
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
            
            print(f"Number of cells from conc {conc} before filtering:", counts.shape[0])

            # Filter for fractionMT
            fractionMT_filter = col_data['fractionMT'] <= 0.2

            # Calculate percentiles for n_gene and n_umi
            n_gene_percentile = np.percentile(col_data['n_gene'], percentile)
            n_umi_percentile = np.percentile(col_data['n_umi'], percentile)

            # Create filters for n_gene and n_umi
            n_gene_filter = col_data['n_gene'] > n_gene_percentile
            n_umi_filter = col_data['n_umi'] > n_umi_percentile

            # Cell type filter
            cell_type_filter = np.isin(col_data['celltype_major'], celltypes_to_keep)

            # Combine all filters
            combined_filter = fractionMT_filter & n_gene_filter & n_umi_filter & cell_type_filter
            
            # Convert boolean mask to indices
            indices = np.where(combined_filter)[0]

            # Apply filter to counts dataset using the indices
            filtered_counts = counts[indices, :]

            # Apply the same filter to colData dataset
            filtered_col_data = col_data[indices]

            print(f"Number of cells from conc {conc} after filtering:", filtered_counts.shape[0])

            # Identify ADT_ fields excluding ADT_Hashtag fields
            adt_fields = [field for field in col_data.dtype.names if field.startswith('ADT_') and 'Hashtag' not in field and 'barcodes' not in field]
            print("entry of first ADT field:", col_data[adt_fields[0]][0])
            # Extract these fields for the filtered cells
            adt_data = np.array([filtered_col_data[field] for field in adt_fields]).T
            no_ADT_fields = adt_data.shape[1]
            print("shape of adt_data:", adt_data.shape)
            # Concatenate these fields with the filtered count dataset
            # Assuming filtered_counts and adt_data are numpy arrays
            combined_data = np.hstack((adt_data, filtered_counts))
            n_cells = filtered_counts.shape[0]
            conc_data = np.full(n_cells, conc)  # Create an array filled with the current concentration

            # Create a DataFrame for the metadata
            metadata_df = pd.DataFrame({
                'cycle_phase': filtered_col_data['cycle_phase'],
                'celltype_major': filtered_col_data['celltype_major'],
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


def highly_variable_genes(AB_log_transformed_gene_data, lectin_log_transformed_gene_data, common_genes, OUTDIR):
    # Combine the two datasets
    combined_gene_data = np.concatenate([AB_log_transformed_gene_data, lectin_log_transformed_gene_data], axis=0)

    # Calculate variance for each gene
    gene_variances = np.var(combined_gene_data, axis=0)

    # Sort genes by variance
    sorted_indices = np.argsort(gene_variances)[::-1]
    sorted_genes = [common_genes[i] for i in sorted_indices]
    sorted_variances = gene_variances[sorted_indices]

    # Prepare the result in the desired format
    hvg_results = [(idx, gene, var) for idx, gene, var in zip(sorted_indices, sorted_genes, sorted_variances)]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_variances[:500])), sorted_variances[:500])
    plt.title('Top 500 Highly Variable Genes')
    plt.xlabel('Genes')
    plt.ylabel('Variance')
    plt.axvline(x=500, color='red', linestyle='--')  # Assuming you want to visualize the top 500 genes
    plt.savefig(OUTDIR / 'highly_variable_genes.png')

    return hvg_results



def only_highly_variable_genes_old(adata, non_gene_features):
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

def CLR_across_cells(data, pseudocount=0.01):
    adjusted_data = data + pseudocount
    geometric_means = np.exp(np.mean(np.log(adjusted_data), axis=0))
    normalized_data = np.log(adjusted_data) - np.log(geometric_means)
    return normalized_data

def CLR_across_features(data, pseudocount=0.01):
    adjusted_data = data + pseudocount
    geometric_means = np.exp(np.mean(np.log(adjusted_data), axis=1, keepdims=True))
    normalized_data = np.log(adjusted_data) - np.log(geometric_means)

    return normalized_data


def normalize_to_median_sample(data, sample_size=100):
    # Randomly sample rows and compute their median sum
    random_sample = data[np.random.choice(data.shape[0], sample_size, replace=False), :]
    median_sum = np.median(random_sample.sum(axis=1))
    
    # Normalize data
    row_sums = data.sum(axis=1)
    if row_sums[row_sums == 0].shape[0] > 0:
        print("row sums with zero: ", row_sums[row_sums == 0])
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized_data = (data.T / row_sums * median_sum).T
    return normalized_data


def filter_genes_for_analysis_old(celltype_markers, hvg_results, AB_log_transformed_gene_data, lectin_log_transformed_gene_data, common_genes, only_marker = False):
    # Extract top 40 cell type marker genes. Including additional marker genes doesn't seem to improve cell type classification with a random forest classifier.
    top_celltype_markers = set([gene for _, gene, _ in celltype_markers[:40]])

    # Initialize variables for HVGs
    top_hvg = set([gene for _, gene, _ in hvg_results[:200]])
    additional_hvg_needed = len(top_celltype_markers.intersection(top_hvg))


    # Add HVGs until 
    # there are 240 unique genes in total
    while_entered = 0
    while additional_hvg_needed > 0:
        if while_entered == 0:
            print(f"{additional_hvg_needed} of the 200 most highly variable genes were also in the top 40 cell type marker genes.")
            while_entered = 1
        top_hvg = set([gene for _, gene, _ in hvg_results[:200 + additional_hvg_needed]]).difference(top_celltype_markers)
        additional_hvg_needed = len(top_celltype_markers.intersection(top_hvg))

    # Combine the two sets to get the final list of genes
    final_gene_set = top_celltype_markers.union(top_hvg)
    print("length of final gene set: ", len(final_gene_set))
    #assert len(final_gene_set) == 240, "Final gene set does not contain 240 unique genes."

    # Get indices of the final gene set in common_genes
    final_gene_indices = [common_genes.index(gene) for gene in final_gene_set]

    # Filter the datasets
    filtered_AB_data = AB_log_transformed_gene_data[:, final_gene_indices]
    filtered_lectin_data = lectin_log_transformed_gene_data[:, final_gene_indices]


    if only_marker:
        return 
    else:
        return filtered_AB_data, filtered_lectin_data

def filter_genes_for_analysis(celltype_markers, hvg_results, AB_log_transformed_gene_data, lectin_log_transformed_gene_data, common_genes, only_markers=False):
    # Extract top 40 cell type marker genes.
    top_celltype_markers = set([gene for _, gene, _ in celltype_markers[:40]])

    if only_markers:
        # If only markers are required, filter datasets for these markers and return
        final_gene_indices = [common_genes.index(gene) for gene in top_celltype_markers]
        filtered_AB_data = AB_log_transformed_gene_data[:, final_gene_indices]
        filtered_lectin_data = lectin_log_transformed_gene_data[:, final_gene_indices]
        return filtered_AB_data, filtered_lectin_data
    else:
        # Include top 200 highly variable genes, excluding those already in top cell type markers
        top_hvg = set([gene for _, gene, _ in hvg_results[:200]])
        additional_hvg_needed = 200 - len(top_hvg.difference(top_celltype_markers))

        # Add HVGs until there are 240 unique genes in total
        while additional_hvg_needed > 0:
            top_hvg = set([gene for _, gene, _ in hvg_results[:200 + additional_hvg_needed]]).difference(top_celltype_markers)
            additional_hvg_needed = 240 - len(top_celltype_markers.union(top_hvg))

        # Combine the two sets to get the final list of genes
        final_gene_set = top_celltype_markers.union(top_hvg)
        print("Length of final gene set: ", len(final_gene_set))
        
        # Get indices of the final gene set in common_genes
        final_gene_indices = [common_genes.index(gene) for gene in final_gene_set]

        # Filter the datasets
        filtered_AB_data = AB_log_transformed_gene_data[:, final_gene_indices]
        filtered_lectin_data = lectin_log_transformed_gene_data[:, final_gene_indices]

        return filtered_AB_data, filtered_lectin_data

def save_to_anndata(data_matrix, metadata, file_path):
    # Create an AnnData object
    adata = ad.AnnData(X=data_matrix, obs=metadata)

    # Save the AnnData object to an .h5 file
    adata.write(file_path)
    print(f"Data saved to {file_path}")