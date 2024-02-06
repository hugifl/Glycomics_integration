############################################################################################################
# This script is used to prepare the data for the SEURAT model.
# The data is filtered and gene expression features are matched across datasets.
# The data is exported to be used in R. 


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

OUTIDR = Path('/cluster/scratch/hugifl/glycomics_for_seurat_2')

# ----------------------------------------------------------- Parameters ----------------------------------------------------------
AB_viable_conc = [0.1, 0.25, 0.5, 1, 2]
lectin_viable_conc = [0.1, 0.5, 1, 2] # there is 0.0 
celltypes_major_to_keep = [1,3,4,5,6,7]
percentile = 10
# ----------------------------------------------------------- finding set of common genes ----------------------------------------------------------
# In case the AB and the lectin datasets and the different titration concentrations have different sets of genes, 
# we find the set of common genes and reduce the datasets to this set. The common genes are returned and all the files are processed and saved in the OUTDIR directory.


common_genes = reduce_to_common_genes(AB_viable_conc, 'AB', lectin_viable_conc, 'lectin', OUTIDR)
#test_reduced_genes(AB_viable_conc, 'AB', lectin_viable_conc, 'lectin', OUTIDR)

# ----------------------------------------------------------- filtering and combining data ----------------------------------------------------------
# Filtering out cells that are in the lowest 10-th percentile of UMI counts and/or detected genes and cells with MT fraction > 0.2.
# Combining the all titration concentrations except the highest one (4 mikrog/ml) into one dataset. 

# Antibody dataset
AB_final_counts, AB_final_counts_no_gex, AB_final_metadata, AB_no_ADT_fields = load_filter_combine_data(AB_viable_conc, dataset_type='AB', OUTDIR=OUTIDR, celltypes_to_keep=celltypes_major_to_keep,  percentile = percentile)
lectin_final_counts, lectin_final_counts_no_gex, lectin_final_metadata, lectin_no_ADT_fields = load_filter_combine_data(lectin_viable_conc, dataset_type='lectin', OUTDIR=OUTIDR, celltypes_to_keep=celltypes_major_to_keep,  percentile = percentile)

AB_ADT_data = AB_final_counts[:, :AB_no_ADT_fields]
AB_gene_data = AB_final_counts[:, AB_no_ADT_fields:]

lectin_ADT_data = lectin_final_counts[:, :lectin_no_ADT_fields]
lectin_gene_data = lectin_final_counts[:, lectin_no_ADT_fields:]


# --------------------------------------------------------------------- Save datasets -------------------------------------------------------------------

AB_file_path_genes = OUTIDR / "AB_gex.h5ad"  
lectin_file_path_genes =  OUTIDR / "lectin_gex.h5ad"   
AB_file_path_adt = OUTIDR / "AB_adt.h5ad"  
lectin_file_path_adt =  OUTIDR / "lectin_adt.h5ad"  

save_to_anndata(AB_gene_data, AB_final_metadata, AB_file_path_genes)
save_to_anndata(lectin_gene_data, lectin_final_metadata, lectin_file_path_genes)
save_to_anndata(AB_ADT_data, AB_final_metadata, AB_file_path_adt)
save_to_anndata(lectin_ADT_data, lectin_final_metadata, lectin_file_path_adt)

    