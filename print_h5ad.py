import anndata


# Load the .h5ad file
adata = anndata.read_h5ad('/cluster/scratch/hugifl/4_glycomics_5/2_integration_iteration_8_AB/codes.h5ad')

# Print the components and their shapes
print("AnnData object structure:\n")
print(f"Observations (rows): {adata.obs.shape}")
print(f"Variables (columns): {adata.var.shape}")
print(f"Variables: {adata.var}")
print(f"Main data matrix (X): {adata.X.shape}")

if adata.raw:
    print(f"Raw data matrix: {adata.raw.X.shape}")

if adata.layers:
    print("\nLayers:")
    for layer in adata.layers.keys():
        print(f"Layer '{layer}' shape: {adata.layers[layer].shape}")

if adata.obsm:
    print("\nObsm (Observation-level multidimensional annotations):")
    for key in adata.obsm.keys():
        print(f"'{key}' shape: {adata.obsm[key].shape}")

if adata.varm:
    print("\nVarm (Variable-level multidimensional annotations):")
    for key in adata.varm.keys():
        print(f"'{key}' shape: {adata.varm[key].shape}")

if adata.obsp:
    print("\nObsp (Pairwise annotations of observations):")
    for key in adata.obsp.keys():
        print(f"'{key}' shape: {adata.obsp[key].shape}")

if adata.varp:
    print("\nVarp (Pairwise annotations of variables):")
    for key in adata.varp.keys():
        print(f"'{key}' shape: {adata.varp[key].shape}")