
import os
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy

normalization_method = 'CLR_feature' # 'library_size' or 'CLR_cell' or 'CLR_feature'
dir = f'/cluster/scratch/hugifl/9_glycomics_6_3_c2_{normalization_method}'

outdir = f'/cluster/home/hugifl/scim/feature_count_distributions_{normalization_method}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def load_h5ad_to_df(filepath):
    """Load an .h5ad file and return its .X attribute as a DataFrame."""
    adata = ad.read_h5ad(filepath)
    # Convert the .X (data matrix) to a DataFrame, assuming it's stored as an ndarray or similar
    # For sparse matrices, you might need to convert them to dense: .toarray() or .todense()
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    return df

#def plot_feature_distributions(df, title):
#    """Plot the distribution of counts for each feature in the DataFrame."""
#    # Limit the number of plots for practicality; adjust as needed
#    num_plots = min(len(df.columns), 100)  
#    fig, axes = plt.subplots(nrows=num_plots, figsize=(10, 2*num_plots))
#    for i, column in enumerate(df.columns[:num_plots]):
#        sns.histplot(df[column], ax=axes[i], bins=30, kde=True)
#        axes[i].set_title(f'Distribution of {column}')
#        axes[i].set_xlabel('Counts')
#        axes[i].set_ylabel('Frequency')
#    plt.tight_layout()
#    plt.suptitle(title, y=1.02, fontsize=16)
#    plt.savefig(f'{outdir}/{title}.png')
#
# Example usage
ab_df = load_h5ad_to_df('/cluster/scratch/hugifl/9_glycomics_6_3_c2_library/AB.h5ad')
lectin_df = load_h5ad_to_df('/cluster/scratch/hugifl/9_glycomics_6_3_c2_library/lectin.h5ad')

# Plot the distributions
#plot_feature_distributions(ab_df, 'Surface Proteins in AB.h5ad')
#plot_feature_distributions(lectin_df, 'Glycans in lectin.h5ad')


def plot_feature_distributions_h5ad(filepath, feature_names, title, outdir):
    """Plot the distribution of counts for each feature in an h5ad file, with up to 4 plots per row."""
    # Load the .h5ad file
    adata = ad.read_h5ad(filepath)
    
    # Check if the dataset is stored as sparse matrix and convert to dense if necessary
    if isinstance(adata.X, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        data = adata.X.toarray()
    else:
        data = adata.X
    
    # Calculate the number of rows needed for the subplots
    nrows = (len(feature_names) + 3) // 4  # +3 for rounding up the division
    
    # Plot the distributions
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(20, 2*nrows))  # Adjusted figsize
    axes = axes.flatten()  # Flatten the axes array for easy iteration
    
    for i, feature in enumerate(feature_names):
        sns.histplot(data[:, i], ax=axes[i], bins=30, kde=True)
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel('Counts')
        axes[i].set_ylabel('Frequency')
        
    # Hide the unused axes if the number of features is not a multiple of 4
    for j in range(i + 1, nrows * 4):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(f'{outdir}/{title}.png')

# Define the feature names
feature_names = (
    'ADT_AAL', 'ADT_PNA', 'ADT_UEA.II', 'ADT_LcH', 'ADT_PHA_E', 'ADT_SNA', 
    'ADT_VVA', 'ADT_PSA', 'ADT_ECA', 'ADT_HPA', 'ADT_Jacalin', 'ADT_RCA', 
    'ADT_WGA', 'ADT_UEA.I', 'ADT_ConA', 'ADT_AOL'
)

filepath = dir + '/lectin.h5ad'
plot_feature_distributions_h5ad(filepath, feature_names, 'Glycan features', outdir)

feature_names = ('ADT_SSEA3_IgM', 'ADT_CA19.9', 'ADT_BloodGroupAB_Ag', 'ADT_SSEA4', 'ADT_GD2', 'ADT_GD3',
                  'ADT_Globo_H', 'ADT_CD17', 'ADT_SSEA3_IgG', 'ADT_CD77', 'ADT_MUC16', 'ADT_MUC1', 'ADT_Siglec9', 
                  'ADT_Siglec8', 'ADT_CD370', 'ADT_CD207', 'ADT_CD325', 'ADT_CD144', 'ADT_CD309', 'ADT_CD62E', 'ADT_CD106', 'ADT_CD224', 
                  'ADT_EGFR', 'ADT_CD140a', 'ADT_CD140b', 'ADT_CD193', 'ADT_Notch', 'ADT_XCR1', 'ADT_CD357', 'ADT_KLRG1')


filepath = dir + '/AB.h5ad'
plot_feature_distributions_h5ad(filepath, feature_names, 'Protein features', outdir)
