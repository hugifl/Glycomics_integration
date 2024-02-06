############################################################################################################
# This script loads the WNN graph matrix and computes the UMAP embeddings for the AB and lectin datasets respectively.


import scipy.sparse
import umap.umap_ as umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from utils_data_prep import save_to_anndata


def plot_umap(umap_df, plot_filename, color):
    
    point_size = 5  
    transparency = 0.5 

    plt.figure(figsize=(10, 8))  

    for celltype in sorted(umap_df[color].unique()):
        subset = umap_df[umap_df[color] == celltype]
        plt.scatter(subset['UMAP1'], subset['UMAP2'], label=celltype, alpha=transparency, s=point_size)

        centroid = subset[['UMAP1', 'UMAP2']].mean()
        plt.text(centroid['UMAP1'], centroid['UMAP2'], celltype, fontsize=9, ha='center', va='center', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    plt.legend()
    plt.title("UMAP Plot Colored by Cell Type")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")

    plt.savefig(plot_filename)
    plt.show()
    plt.show()


outdir = '/cluster/home/hugifl/scim/plots/'
outdatadir = '/cluster/scratch/hugifl/20_glycomics/'
plot_filename = outdir + 'wnn_graph_umap3.png'
datadir = '/cluster/home/hugifl/scim/seurat_data/'

umap_components = 10
n_neighbors = 15    
min_dist = 0.8     
learning_rate = 1.0  
n_epochs = None  
spread = 1
local_connectivity = 1
repulsion_strength = 1



rna_pca_embeddings = pd.read_csv(datadir+"rna_pca_embeddings_AB.csv", index_col=0)
adt_pca_embeddings = pd.read_csv(datadir+"adt_pca_embeddings_AB.csv", index_col=0)
celltype_major_AB = pd.read_csv(datadir+"celltype_major_AB.csv")
celltype_major_lectin = pd.read_csv(datadir+"celltype_major_lectin.csv")
celltype_final_AB = pd.read_csv(datadir+"celltype_final_AB.csv")
celltype_final_lectin = pd.read_csv(datadir+"celltype_final_lectin.csv")

# -------------------------------- AB --------------------------------
wnn_graph_df_AB = pd.read_csv(datadir+"wnn_graph_matrix_AB.csv", index_col=0)

# Convert DataFrame to a Sparse Matrix
wnn_graph_sparse_AB = scipy.sparse.csr_matrix(wnn_graph_df_AB.values)
ones_matrix = sp.csr_matrix(wnn_graph_sparse_AB.shape)
ones_matrix[:] = 1

# Subtract the WNN graph from the ones matrix
wnn_graph_distance_AB = ones_matrix - wnn_graph_sparse_AB

# Set the diagonal elements to zero
wnn_graph_distance_AB.setdiag(0)
wnn_graph_distance_AB.eliminate_zeros()
umap_model_AB = umap.UMAP(
    n_components=umap_components,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    learning_rate=learning_rate,
    n_epochs=n_epochs,
    metric='precomputed',
    random_state=42,
    spread=spread,
    local_connectivity=local_connectivity,
    repulsion_strength=repulsion_strength
)
umap_embeddings_AB = umap_model_AB.fit_transform(wnn_graph_distance_AB)

# Create a DataFrame for the UMAP embeddings
umap_df_AB = pd.DataFrame(umap_embeddings_AB, columns=[f'UMAP{i+1}' for i in range(umap_components)])
umap_df_AB['celltype_major'] = celltype_major_AB['x']
umap_df_AB['celltype_final'] = celltype_final_AB['x']

#plot_filename = outdir + 'wnn_graph_umap_AB_celltype_major4.png'
#plot_umap(umap_df_AB, plot_filename,'celltype_major')
#plot_filename = outdir + 'wnn_graph_umap_AB_celltype_final4.png'
#plot_umap(umap_df_AB, plot_filename,'celltype_final')

# Save to anndata
metadata = umap_df_AB[['celltype_major', 'celltype_final']]
data_matrix = umap_df_AB.drop(['celltype_major', 'celltype_final'], axis=1)
anndata_file_path = outdatadir + 'AB.h5ad'
save_to_anndata(data_matrix, metadata, anndata_file_path)

# -------------------------------- Lectin --------------------------------

wnn_graph_df_lectin = pd.read_csv(datadir+"wnn_graph_matrix_lectin.csv", index_col=0)

# Convert DataFrame to a Sparse Matrix
wnn_graph_sparse_lectin = scipy.sparse.csr_matrix(wnn_graph_df_lectin.values)
ones_matrix = sp.csr_matrix(wnn_graph_sparse_lectin.shape)
ones_matrix[:] = 1

# Subtract the WNN graph from the ones matrix
wnn_graph_distance_lectin = ones_matrix - wnn_graph_sparse_lectin

# Set the diagonal elements to zero
wnn_graph_distance_lectin.setdiag(0)
wnn_graph_distance_lectin.eliminate_zeros()
umap_model_lectin = umap.UMAP(
    n_components=umap_components,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    learning_rate=learning_rate,
    n_epochs=n_epochs,
    metric='precomputed',
    random_state=42
)
umap_embeddings_lectin = umap_model_lectin.fit_transform(wnn_graph_distance_lectin)


# Create a DataFrame for the UMAP embeddings
umap_df_lectin = pd.DataFrame(umap_embeddings_lectin, columns=[f'UMAP{i+1}' for i in range(umap_components)])
umap_df_lectin['celltype_major'] = celltype_major_lectin['x']
umap_df_lectin['celltype_final'] = celltype_final_lectin['x']

## Save to anndata
metadata = umap_df_lectin[['celltype_major', 'celltype_final']]
data_matrix = umap_df_lectin.drop(['celltype_major', 'celltype_final'], axis=1)
anndata_file_path = outdatadir + 'lectin.h5ad'
save_to_anndata(data_matrix, metadata, anndata_file_path)
#plot_filename = outdir + 'wnn_graph_umap_lectin_major4.png'
#plot_umap(umap_df_lectin, plot_filename, 'celltype_major')
#plot_filename = outdir + 'wnn_graph_umap_lectin_final4.png'
#plot_umap(umap_df_lectin, plot_filename, 'celltype_final')

