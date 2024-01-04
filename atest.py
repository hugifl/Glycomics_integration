import anndata

# Path to your .h5ad file
file_path = '/cluster/scratch/hugifl/cytof_scRNA/cytof.h5ad'

# Load the .h5ad file
evald = anndata.read(file_path)

print("columns of obs: ", evald.obs.columns)
# Check if 'branch' is in the .obs or .var attributes
if 'branch' in evald.obs.columns:
    print("'branch' is a column in evald.obs")
else:
    print("'branch' is NOT a column in evald.obs")

if 'branch' in evald.var_names:
    print("'branch' is a column in evald.var")
else:
    print("'branch' is NOT a column in evald.var")