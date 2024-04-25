############################################################################################################
# This script is used to prepare the data for the SCIM model. 
# Crucial steps include:
# - Filtering out low quality cells.
# - Matching the gene expression features from the different datasets.
# - Normalizing and scaling the data.
# - Feature selection.

import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import anndata
from utils_data_prep import (load_filter_combine_data, highly_variable_genes, 
                             normalize_to_median_sample, reduce_to_common_genes, 
                             test_reduced_genes, find_celltype_marker_genes, 
                             filter_genes_for_analysis, save_to_anndata, CLR_across_cells, CLR_across_features)
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler

OUTIDR = Path('/cluster/scratch/hugifl/glycomics_c2_top_3')


# ----------------------------------------------------------- Parameters ----------------------------------------------------------
AB_viable_conc = [2] # [0.1, 0.25, 0.5, 1, 2]
lectin_viable_conc = [2] # there is 0.0  [0.1, 0.5, 1, 2]
celltypes_major_to_keep = [4,5,1] # [4,5,1,3,6,7]
all_features = False # if True, all features are used, if False, only either expression or ADT features are used
only_gex = False # if True, all features are used, if False, only either expression or ADT features are used
no_gex = True 
only_markers = True   # if True, only the marker genes are used, if False, marker genes + highly variable genes are used
scaling = False
log = True #set to true usually
percentile = 10
ADT_normalization = 'library_size' # CLR_across_cells, CLR_across_features, library_size

# ----------------------------------------------------------- finding set of common genes ----------------------------------------------------------
# In case the AB and the lectin datasets and the different titration concentrations have different sets of genes, 
# we find the set of common genes and reduce the datasets to this set. The common genes are returned and all the files are processed and saved in the OUTDIR directory.


common_genes = reduce_to_common_genes(AB_viable_conc, 'AB', lectin_viable_conc, 'lectin', OUTIDR)
#test_reduced_genes(AB_viable_conc, 'AB', lectin_viable_conc, 'lectin', OUTIDR)

# ----------------------------------------------------------- filtering and combining data ----------------------------------------------------------
# Filtering out cells that are in the lowest 10-th percentile of UMI counts and/or detected genes and cells with MT fraction > 0.2.
# Combining the all titration concentrations except the highest one (4 mikrog/ml) into one dataset. 

# Antibody dataset
AB_final_counts, AB_final_counts_no_gex, AB_final_metadata, AB_no_ADT_fields = load_filter_combine_data(AB_viable_conc, dataset_type='AB', OUTDIR=OUTIDR, celltypes_to_keep=celltypes_major_to_keep, percentile = percentile)
lectin_final_counts, lectin_final_counts_no_gex, lectin_final_metadata, lectin_no_ADT_fields = load_filter_combine_data(lectin_viable_conc, dataset_type='lectin', OUTDIR=OUTIDR, celltypes_to_keep=celltypes_major_to_keep,  percentile = percentile)

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
lectin_normalized_gene_data = normalize_to_median_sample(lectin_gene_data, sample_size=500)

if ADT_normalization == 'library_size':
    AB_normalized_ADT_data = normalize_to_median_sample(AB_ADT_data, sample_size=500)
    lectin_normalized_ADT_data = normalize_to_median_sample(lectin_ADT_data, sample_size=500)

if ADT_normalization == 'CLR_across_cells': 
    AB_normalized_ADT_data = CLR_across_cells(AB_ADT_data)
    lectin_normalized_ADT_data = CLR_across_cells(lectin_ADT_data)

if ADT_normalization == 'CLR_across_features':
    AB_normalized_ADT_data = CLR_across_features(AB_ADT_data)
    lectin_normalized_ADT_data = CLR_across_features(lectin_ADT_data)

# Log transform
if log and ADT_normalization == 'library_size':
    AB_log_transformed_gene_data = np.log1p(AB_normalized_gene_data)
    AB_log_transformed_ADT_data = np.log1p(AB_normalized_ADT_data)

    lectin_log_transformed_gene_data = np.log1p(lectin_normalized_gene_data)
    lectin_log_transformed_ADT_data = np.log1p(lectin_normalized_ADT_data)
else:
    AB_log_transformed_gene_data = AB_normalized_gene_data
    AB_log_transformed_ADT_data = AB_normalized_ADT_data

    lectin_log_transformed_gene_data = lectin_normalized_gene_data
    lectin_log_transformed_ADT_data = lectin_normalized_ADT_data

# ----------------------------------------------------------- Reduce gene expression features ----------------------------------------------------------
# To reduce the number of features, the top 40 most important genes for classifying the cell types using a random forest classifier are selected.
# Additionally the top 200 most variable genes are selected to get a final set of 240 gene features.

celltype_markers, marker_names = find_celltype_marker_genes(AB_final_metadata, lectin_final_metadata, AB_log_transformed_gene_data, lectin_log_transformed_gene_data, common_genes, OUTIDR)
print("names of first 40 markers: ", marker_names[:20])

hvg_results = highly_variable_genes(AB_log_transformed_gene_data, lectin_log_transformed_gene_data, common_genes, OUTIDR)

filtered_AB_data, filtered_lectin_data = filter_genes_for_analysis(celltype_markers, hvg_results, AB_log_transformed_gene_data, lectin_log_transformed_gene_data, common_genes, only_markers)

# ------------------------------------------------------------------- Combine datasets -----------------------------------------------------------------
# The lectin and AB final datasets are produced by concatenating the gene expression and AB/lectin feature counts.
# The features are scaled to the 0-1 range using the MinMaxScaler from scikit-learn to ensure that the features are on the same scale.

# Combine gene expression and protein abundance
AB_combined_data = np.concatenate((AB_log_transformed_ADT_data, filtered_AB_data), axis=1)
lectin_combined_data = np.concatenate((lectin_log_transformed_ADT_data, filtered_lectin_data), axis=1)

# Min-Max scale the combined data
if scaling:
    AB_scaler = MinMaxScaler(feature_range=(0, 1))
    AB_combined_data = AB_scaler.fit_transform(AB_combined_data)

    lectin_scaler = MinMaxScaler(feature_range=(0, 1))
    lectin_combined_data = lectin_scaler.fit_transform(lectin_combined_data)

# --------------------------------------------------------------------- Save datasets -------------------------------------------------------------------

AB_file_path = OUTIDR / "AB.h5ad"  
lectin_file_path =  OUTIDR / "lectin.h5ad"   

print("marker names: ", marker_names)
print("first 40 markers: ", marker_names[:40])
# saving the marker names
marker_names_df = pd.DataFrame(marker_names)
marker_names_df.to_csv(OUTIDR / "marker_names.csv")

if all_features:
    save_to_anndata(AB_combined_data, AB_final_metadata, AB_file_path)
    save_to_anndata(lectin_combined_data, lectin_final_metadata, lectin_file_path)
if only_gex:
    save_to_anndata(filtered_AB_data, AB_final_metadata, AB_file_path)
    save_to_anndata(filtered_lectin_data, lectin_final_metadata, lectin_file_path)
if no_gex:
    save_to_anndata(AB_log_transformed_ADT_data, AB_final_metadata, AB_file_path)
    save_to_anndata(lectin_log_transformed_ADT_data, lectin_final_metadata, lectin_file_path)


