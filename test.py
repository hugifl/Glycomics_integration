import anndata

#print("simulated")
## Load the .h5ad file
#adata = anndata.read("/cluster/scratch/hugifl/SCIM_data_simulated_2/source.h5ad")
#
## Print the columns of the .obs dataframe
#
## Check the dimensions of the X matrix
#print("Dimensions of X:", adata.X.shape)
#print("Average number of zeros per row:", (adata.X == 0).sum(axis=1).mean())
#print("Mean value of X:", adata.X.mean())
#print("Standard deviation of X:", adata.X.std())
#print("Minimum value of X:", adata.X.min())
#print("Maximum value of X:", adata.X.max())
#mean_per_feature = adata.X.mean(axis=0)
#print("avg mean value per feature:", mean_per_feature.mean())
#print("std mean value per feature:", mean_per_feature.std())
#print("observations : ", adata.obs.columns)
#
print("lectin")

adata = anndata.read("/cluster/scratch/hugifl/glycomics_4_less_KL_more_units_in_lat/lectin.h5ad")

# Print the columns of the .obs dataframe

# Check the dimensions of the X matrix
print("X head: ", adata.X[2,:16].mean())
print("X head: ", adata.X[2,16:].mean())
print("Dimensions of X:", adata.X.shape)
print("Average number of zeros per row:", (adata.X == 0).sum(axis=1).mean())
print("Mean value of X:", adata.X.mean())
print("Standard deviation of X:", adata.X.std())
print("Minimum value of X:", adata.X.min())
print("Maximum value of X:", adata.X.max())
mean_per_feature = adata.X.mean(axis=0)
print("avg mean value per feature:", mean_per_feature.mean())
print("std mean value per feature:", mean_per_feature.std())
print("observations : ", adata.obs.columns)
print("unique labels major: ", adata.obs['celltype_major'].unique())
print("unique labels final: ", adata.obs['celltype_final'].unique())

#adata = anndata.read("/cluster/scratch/hugifl/TitrationII/AB.h5ad")
#
## Print the columns of the .obs dataframe
#
## Check the dimensions of the X matrix
#print("Dimensions of X:", adata.X.shape)
#print("Average number of zeros per row:", (adata.X == 0).sum(axis=1).mean())
#print("Mean value of X:", adata.X.mean())
#print("Standard deviation of X:", adata.X.std())
#print("Minimum value of X:", adata.X.min())
#print("Maximum value of X:", adata.X.max())
#mean_per_feature = adata.X.mean(axis=0)
#print("avg mean value per feature:", mean_per_feature.mean())
#print("std mean value per feature:", mean_per_feature.std())
#

#print("cytof")
#
#adata = anndata.read("/cluster/scratch/hugifl/cytof_scRNA/cytof.h5ad")
#
## Print the columns of the .obs dataframe
#
## Check the dimensions of the X matrix
#print("Dimensions of X:", adata.X.shape[1])
#print("Average number of zeros per row:", (adata.X == 0).sum(axis=1).mean())
#print("Mean value of X:", adata.X.mean())
#print("Standard deviation of X:", adata.X.std())
#print("Minimum value of X:", adata.X.min())
#print("Maximum value of X:", adata.X.max())
#mean_per_feature = adata.X.mean(axis=0)
#print("avg mean value per feature:", mean_per_feature.mean())
#print("std mean value per feature:", mean_per_feature.std())
#
#print("scrna")
#
#adata = anndata.read("/cluster/scratch/hugifl/cytof_scRNA/scrna.h5ad")
#
## Print the columns of the .obs dataframe
#
## Check the dimensions of the X matrix
#print("Dimensions of X:", adata.X.shape)
#print("Average number of zeros per row:", (adata.X == 0).sum(axis=1).mean())
#print("Mean value of X:", adata.X.mean())
#print("Standard deviation of X:", adata.X.std())
#print("Minimum value of X:", adata.X.min())
#print("Maximum value of X:", adata.X.max())
#mean_per_feature = adata.X.mean(axis=0)
#print("avg mean value per feature:", mean_per_feature.mean())
#print("std mean value per feature:", mean_per_feature.std())
#
#print("observations : ", adata.obs.columns)
