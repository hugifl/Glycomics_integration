import anndata

# Path to your .h5ad file
file_path = '/cluster/scratch/hugifl/cytof_scRNA/scrna.h5ad'
file_path_2 = '/cluster/scratch/hugifl/glycomics/AB.h5ad'

def calculate_celltype_percentages_from_files(file_paths, label):
    percentages = {}
    counts_dict = {}
    for tech, file_path in file_paths.items():
        # Load the data from the .h5ad file
        data = anndata.read(file_path)

        # Count occurrences of each cell type
        counts = data.obs[label].value_counts()
        counts_dict[tech] = counts
        # Calculate percentages
        total_cells = len(data)
        percentages[tech] = (counts / total_cells * 100).to_dict()

    return percentages, counts_dict

# Usage example
file_paths = {
    'lectin': '/cluster/scratch/hugifl/glycomics/lectin.h5ad',
    'AB': '/cluster/scratch/hugifl/glycomics/AB.h5ad'
}
LABEL = 'celltype_major'  # Ensure this is the correct label
percentages, counts = calculate_celltype_percentages_from_files(file_paths, LABEL)
for tech, pct in percentages.items():
    print(f"Percentages for {tech}:")
    for cell_type, percentage in pct.items():
        print(f"  Cell Type {cell_type}: {percentage:.2f}%")

for tech, ct in counts.items():
    print(f"Counts for {tech}:")
    for cell_type, count in ct.items():
        print(f"  Cell Type {cell_type}: {count:.2f}")
## Load the .h5ad file
#data_paper = anndata.read(file_path)
#data_new = anndata.read(file_path_2)
#
#print("head of obs paper: ", data_paper.obs.head())
#print("head of obs new: ", data_new.obs.head())
#
#print("feature number paper:", data_paper.X.shape[1])
#print("feature number new:", data_new.X.shape[1])
#
#print("cells number paper:", data_paper.X.shape[0])
#print("cells number new:", data_new.X.shape[0])
#
#print("head of X paper: ", data_paper.X[5, :100])
#print("max of X paper: ", data_paper.X[5, :].max())
#print("sum of X paper: ", data_paper.X[5, :].sum())
#print("average of X paper: ", data_paper.X[5, :].mean())
#print("head of X new: ", data_new.X[50, 31:50])
#print("max of X new: ", data_new.X[50, 31:].max())
#print("sum of X new: ", data_new.X[50, 31:].sum())
#print("average of X new: ", data_new.X[50, 31:].mean())
#print("average of X new: ", data_new.X[70, 31:].mean())
#print("average of X new: ", data_new.X[100, 31:].mean())