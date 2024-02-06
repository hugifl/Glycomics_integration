# Converting the .h5ad files to .csv files for Seurat (R)

import anndata

# Load the .h5ad file
AB_ADT = anndata.read_h5ad("/cluster/scratch/hugifl/glycomics_for_seurat_2/AB_adt.h5ad")
lectin_ADT = anndata.read_h5ad("/cluster/scratch/hugifl/glycomics_for_seurat_2/lectin_adt.h5ad") 
AB_gex = anndata.read_h5ad("/cluster/scratch/hugifl/glycomics_for_seurat_2/AB_gex.h5ad")
lectin_gex = anndata.read_h5ad("/cluster/scratch/hugifl/glycomics_for_seurat_2/lectin_gex.h5ad") 
# Save the expression matrix (and optionally, other data like metadata)
AB_ADT.to_df().to_csv("/cluster/scratch/hugifl/glycomics_for_seurat_2/AB_ADT_matrix.csv")
# If you have metadata or other annotations you need, save them as well
AB_ADT.obs.to_csv("/cluster/scratch/hugifl/glycomics_for_seurat_2/AB_ADT_metadata.csv")

lectin_ADT.to_df().to_csv("/cluster/scratch/hugifl/glycomics_for_seurat_2/lectin_ADT_matrix.csv")
# If you have metadata or other annotations you need, save them as well
lectin_ADT.obs.to_csv("/cluster/scratch/hugifl/glycomics_for_seurat_2/lectin_ADT_metadata.csv")

AB_gex.to_df().to_csv("/cluster/scratch/hugifl/glycomics_for_seurat_2/AB_gex_matrix.csv")
# If you have metadata or other annotations you need, save them as well
AB_gex.obs.to_csv("/cluster/scratch/hugifl/glycomics_for_seurat_2/AB_gex_metadata.csv")

lectin_gex.to_df().to_csv("/cluster/scratch/hugifl/glycomics_for_seurat_2/lectin_gex_matrix.csv")
# If you have metadata or other annotations you need, save them as well
lectin_gex.obs.to_csv("/cluster/scratch/hugifl/glycomics_for_seurat_2/lectin_gex_metadata.csv")