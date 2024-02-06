# This script is used to construct and plot the UMAP embeddings of the AB and lectin dataset integrated with SEURAT.

import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
from anndata import AnnData
import scanpy as sc
import numpy as np

def annotate_centroids_2(ax, data, label_col, plot_method):
    cell_types = data.obs[label_col].dropna().unique()
    for cell_type in cell_types:
        subset = data[data.obs[label_col] == cell_type]
        if 'X_' + plot_method in subset.obsm and len(subset) > 0:
            centroid = subset.obsm['X_' + plot_method].mean(axis=0)
            ax.text(centroid[0], centroid[1], cell_type, fontsize=16, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
        else:
            print(f"Centroid not calculated for {cell_type}. Subset size: {len(subset)}")

# Paths
outdir = '/cluster/home/hugifl/scim/plots_R/'
datadir = '/cluster/home/hugifl/scim/seurat_data_2/'
dataset = 'concat' # concat or ADT or rna

# Load data
pca_embeddings = pd.read_csv(f"{datadir}{dataset}_PCA_coordinates.csv", index_col=0)
celltype_major = pd.read_csv(f"{datadir}{dataset}_celltype_major.csv")['x'].astype(str)  # Load 'x' column as string
celltype_final = pd.read_csv(f"{datadir}{dataset}_celltype_final.csv")['x'].astype(str)  # Load 'x' column as string

adata = AnnData(X=pca_embeddings)
adata.obs['celltype_major'] = celltype_major.values  # Use values to align with the index of pca_embeddings
adata.obs['celltype_final'] = celltype_final.values  # Use values to align with the index of pca_embeddings

# Run UMAP
umap_model = umap.UMAP(n_components=2, min_dist=0.5)
adata.obsm['X_umap'] = umap_model.fit_transform(adata.X)

# Determine global UMAP limits
umap_limits = {
    'x': (adata.obsm['X_umap'][:, 0].min(), adata.obsm['X_umap'][:, 0].max()),
    'y': (adata.obsm['X_umap'][:, 1].min(), adata.obsm['X_umap'][:, 1].max())
}

# Initial plotting with global UMAP limits
title_fontsize = 20 
axis_label_fontsize = 16  
legend_fontsize = 16  
point_size = 50 

fig, axes = plt.subplots(1, 2, figsize=(24, 8))
# Plot UMAP for celltype_major with global limits
sc.pl.umap(adata, color="celltype_major", ax=axes[0], show=False, size=point_size)
annotate_centroids_2(axes[0], adata, "celltype_major", 'umap')
axes[0].set_xlim(umap_limits['x'])
axes[0].set_ylim(umap_limits['y'])
axes[0].set_title('UMAP colored by Cell Type', fontsize=title_fontsize)
# Plot UMAP for celltype_final with global limits
sc.pl.umap(adata, color="celltype_final", ax=axes[1], show=False, size=point_size)
annotate_centroids_2(axes[1], adata, "celltype_final", 'umap')
axes[1].set_xlim(umap_limits['x'])
axes[1].set_ylim(umap_limits['y'])
axes[1].set_title('UMAP colored by Cell Type', fontsize=title_fontsize)
plt.savefig(f"{outdir}{dataset}_seurat_integrated_concat_umap.png", dpi=300)
plt.close(fig)

# Plotting each cell type individually with consistent scale
unique_cell_types = adata.obs['celltype_major'].unique()
for cell_type in unique_cell_types:
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))
    adata_filtered = adata[adata.obs['celltype_major'] == cell_type].copy()
    # Plot UMAP for the specific cell type with global limits
    sc.pl.umap(adata_filtered, color="celltype_major", ax=axes[0], show=False, size=point_size)
    annotate_centroids_2(axes[0], adata_filtered, "celltype_major", 'umap')
    axes[0].set_xlim(umap_limits['x'])
    axes[0].set_ylim(umap_limits['y'])
    axes[0].set_title(f'UMAP for Cell Type {cell_type}', fontsize=title_fontsize)
    # Plot UMAP for all cell types (for comparison) with global limits
    sc.pl.umap(adata, color="celltype_final", ax=axes[1], show=False, size=point_size)
    annotate_centroids_2(axes[1], adata, "celltype_final", 'umap')
    axes[1].set_xlim(umap_limits['x'])
    axes[1].set_ylim(umap_limits['y'])
    axes[1].set_title('UMAP colored by All Cell Types', fontsize=title_fontsize)
    plt.savefig(f"{outdir}{dataset}_seurat_integrated_concat_umap_{cell_type}.png", dpi=300)
    plt.close(fig)