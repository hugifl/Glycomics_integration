import anndata

# Path to your .h5ad file
file_path = '/cluster/scratch/hugifl/cytof_scRNA/scrna.h5ad'
file_path_2 = '/cluster/scratch/hugifl/glycomics/AB.h5ad'

# Load the .h5ad file
data_paper = anndata.read(file_path)
data_new = anndata.read(file_path_2)

print("head of obs paper: ", data_paper.obs.head())
print("head of obs new: ", data_new.obs.head())

print("feature number paper:", data_paper.X.shape[1])
print("feature number new:", data_new.X.shape[1])

print("cells number paper:", data_paper.X.shape[0])
print("cells number new:", data_new.X.shape[0])

print("head of X paper: ", data_paper.X[5, :100])
print("max of X paper: ", data_paper.X[5, :].max())
print("sum of X paper: ", data_paper.X[5, :].sum())
print("average of X paper: ", data_paper.X[5, :].mean())
print("head of X new: ", data_new.X[50, 31:50])
print("max of X new: ", data_new.X[50, 31:].max())
print("sum of X new: ", data_new.X[50, 31:].sum())
print("average of X new: ", data_new.X[50, 31:].mean())
print("average of X new: ", data_new.X[70, 31:].mean())
print("average of X new: ", data_new.X[100, 31:].mean())