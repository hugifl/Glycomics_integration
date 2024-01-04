import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import anndata
from utils_data_prep import create_dataset, only_highly_variable_genes
import scanpy as sc


OUTIDR = Path('/cluster/scratch/hugifl/TitrationII')

viable_conc_AB = [0.1, 0.25, 0.5, 1, 2]
viable_conc_lectin = [0.0, 0.1, 0.5, 1, 2]

# Antibody dataset
final_counts_AB, final_counts_AB_no_gex, final_metadata_AB, no_ADT_fields = create_dataset(viable_conc_AB, dataset_type='AB', OUTDIR=OUTIDR)
adata_AB = anndata.AnnData(X=final_counts_AB, obs=final_metadata_AB)

# Extract AB counts (first n columns) and gene expression data
AB_data = adata_AB.X[:, :no_ADT_fields]
gene_data = adata_AB.X[:, no_ADT_fields:]

# Normalize protein data
protein_data_adata = anndata.AnnData(AB_data)
sc.pp.normalize_total(protein_data_adata, target_sum=1e4)
normalized_protein_data = protein_data_adata.X

# Normalize gene expression data
gene_data_adata = anndata.AnnData(gene_data)
sc.pp.normalize_total(gene_data_adata, target_sum=1e4)
sc.pp.log1p(gene_data_adata)
normalized_gene_data = gene_data_adata.X

# Concatenate the normalized protein data and gene data
adata_AB.X = np.hstack([normalized_protein_data, normalized_gene_data])

adata_AB_variable = only_highly_variable_genes(adata_AB, no_ADT_fields)

outfile_AB = OUTIDR / 'AB.h5ad'
adata_AB.write(outfile_AB)

outfile_AB = OUTIDR / 'AB_var.h5ad'
adata_AB_variable.write(outfile_AB)

adata_AB_no_gex = anndata.AnnData(X=final_counts_AB_no_gex, obs=final_metadata_AB)
outfile_AB_no_gex = OUTIDR / 'AB_no_gex.h5ad'
adata_AB_no_gex.write(outfile_AB_no_gex)

# Lectin dataset
final_counts_lectin, final_counts_lectin_no_gex, final_metadata_lectin, no_ADT_fields = create_dataset(viable_conc_lectin, dataset_type='lectin', OUTDIR=OUTIDR)
adata_lectin = anndata.AnnData(X=final_counts_lectin, obs=final_metadata_lectin)

# Extract glycan counts (first n columns) and gene expression data
AB_data = adata_lectin.X[:, :no_ADT_fields]
gene_data = adata_lectin.X[:, no_ADT_fields:]

# Normalize protein data
protein_data_adata = anndata.AnnData(AB_data)
sc.pp.normalize_total(protein_data_adata, target_sum=1e4)
normalized_protein_data = protein_data_adata.X

# Normalize gene expression data
gene_data_adata = anndata.AnnData(gene_data)
sc.pp.normalize_total(gene_data_adata, target_sum=1e4)
sc.pp.log1p(gene_data_adata)
normalized_gene_data = gene_data_adata.X

# Concatenate the normalized protein data and gene data
adata_lectin.X = np.hstack([normalized_protein_data, normalized_gene_data])

adata_lectin_variable = only_highly_variable_genes(adata_lectin, no_ADT_fields)

outfile_lectin = OUTIDR / 'lectin.h5ad'
adata_lectin.write(outfile_lectin)

outfile_lectin = OUTIDR / 'lectin_var.h5ad'
adata_lectin_variable.write(outfile_lectin)

adata_lectin_no_gex = anndata.AnnData(X=final_counts_lectin_no_gex, obs=final_metadata_lectin)
outfile_lectin_no_gex = OUTIDR / 'lectin_no_gex.h5ad'
adata_lectin_no_gex.write(outfile_lectin_no_gex)
    

    