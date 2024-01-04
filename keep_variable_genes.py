import pandas as pd
import numpy as np
import scanpy as sc
import anndata

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