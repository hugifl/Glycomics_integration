import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import anndata
from utils_data_prep import (load_filter_combine_data, highly_variable_genes, 
                             normalize_to_median_sample, reduce_to_common_genes, 
                             test_reduced_genes, find_celltype_marker_genes, 
                             filter_genes_for_analysis, save_to_anndata)
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler

OUTIDR = Path('/cluster/scratch/hugifl/glycomics')

AB_viable_conc = [0.1, 0.25, 0.5, 1, 2]
lectin_viable_conc = [0.1, 0.5, 1, 2] # there is 0.0 

# ----------------------------------------------------------- finding set of common genes ----------------------------------------------------------
# In case the AB and the lectin datasets and the different titration concentrations have different sets of genes, 
# we find the set of common genes and reduce the datasets to this set. The common genes are returned and all the files are processed and saved in the OUTDIR directory.


common_genes = reduce_to_common_genes(AB_viable_conc, 'AB', lectin_viable_conc, 'lectin', OUTIDR)
#test_reduced_genes(AB_viable_conc, 'AB', lectin_viable_conc, 'lectin', OUTIDR)

# ----------------------------------------------------------- filtering and combining data ----------------------------------------------------------
# Filtering out cells that are in the lowest 10-th percentile of UMI counts and/or detected genes and cells with MT fraction > 0.2.
# Combining the all titration concentrations except the highest one (4 mikrog/ml) into one dataset. 

# Antibody dataset
AB_final_counts, AB_final_counts_no_gex, AB_final_metadata, AB_no_ADT_fields = load_filter_combine_data(AB_viable_conc, dataset_type='AB', OUTDIR=OUTIDR)
lectin_final_counts, lectin_final_counts_no_gex, lectin_final_metadata, lectin_no_ADT_fields = load_filter_combine_data(lectin_viable_conc, dataset_type='lectin', OUTDIR=OUTIDR)

#adata_AB = anndata.AnnData(X=final_counts_AB, obs=final_metadata_AB)

# ----------------------------------------------------------- normalizing and scaling data ----------------------------------------------------------
# Gene expression and AB/lectin feature counts are library size normalized by scaling them such that each cell has the same cummulative count for all genes/ABs/lectins.
# The cummulative count target for genes / ABs / lectins is set to the median value of a random subset of 500 cells.
# The features are log-transformed and scaled to the 0-1 range using the MinMaxScaler from scikit-learn.

AB_ADT_data = AB_final_counts[:, :AB_no_ADT_fields]
AB_gene_data = AB_final_counts[:, AB_no_ADT_fields:]

lectin_ADT_data = lectin_final_counts[:, :lectin_no_ADT_fields]
lectin_gene_data = lectin_final_counts[:, lectin_no_ADT_fields:]

# Normalize gene_data and AB_data separately
AB_normalized_gene_data = normalize_to_median_sample(AB_gene_data, sample_size=500)
AB_normalized_ADT_data = normalize_to_median_sample(AB_ADT_data, sample_size=500)

lectin_normalized_gene_data = normalize_to_median_sample(lectin_gene_data, sample_size=500)
lectin_normalized_ADT_data = normalize_to_median_sample(lectin_ADT_data, sample_size=500)

# Log transform
AB_log_transformed_gene_data = np.log1p(AB_normalized_gene_data)
AB_log_transformed_ADT_data = np.log1p(AB_normalized_ADT_data)

lectin_log_transformed_gene_data = np.log1p(lectin_normalized_gene_data)
lectin_log_transformed_ADT_data = np.log1p(lectin_normalized_ADT_data)

# ----------------------------------------------------------- Reduce gene expression features ----------------------------------------------------------
# To reduce the number of features, the top 40 most important genes for classifying the cell types using a random forest classifier are selected.
# Additionally the top 200 most variable genes are selected to get a final set of 240 gene features.

celltype_markers = find_celltype_marker_genes(AB_final_metadata, lectin_final_metadata, AB_log_transformed_gene_data, lectin_log_transformed_gene_data, common_genes, OUTIDR)

hvg_results = highly_variable_genes(AB_log_transformed_gene_data, lectin_log_transformed_gene_data, common_genes, OUTIDR)

filtered_AB_data, filtered_lectin_data = filter_genes_for_analysis(celltype_markers, hvg_results, AB_log_transformed_gene_data, lectin_log_transformed_gene_data, common_genes)

# ------------------------------------------------------------------- Combine datasets -----------------------------------------------------------------
# The lectin and AB final datasets are produced by concatenating the gene expression and AB/lectin feature counts.
# The features are scaled to the 0-1 range using the MinMaxScaler from scikit-learn to ensure that the features are on the same scale.

# Combine gene expression and protein abundance
AB_combined_data = np.concatenate((AB_log_transformed_ADT_data, filtered_AB_data), axis=1)
lectin_combined_data = np.concatenate((lectin_log_transformed_ADT_data, filtered_lectin_data), axis=1)
# Min-Max scale the combined data
AB_scaler = MinMaxScaler(feature_range=(0, 1))
AB_scaled_combined_data = AB_scaler.fit_transform(AB_combined_data)

lectin_scaler = MinMaxScaler(feature_range=(0, 1))
lectin_scaled_combined_data = lectin_scaler.fit_transform(lectin_combined_data)

# --------------------------------------------------------------------- Save datasets -------------------------------------------------------------------

AB_file_path = OUTIDR / "AB.h5ad"  
lectin_file_path =  OUTIDR / "lectin.h5ad"   

save_to_anndata(AB_scaled_combined_data, AB_final_metadata, AB_file_path)
save_to_anndata(lectin_scaled_combined_data, lectin_final_metadata, lectin_file_path)



    