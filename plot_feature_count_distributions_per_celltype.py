
import os
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy

dir = '/cluster/scratch/hugifl/glycomics_all_viable'

outdir = '/cluster/home/hugifl/scim/feature_count_distributions_vs_imputed'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def load_h5ad_to_df(filepath):
    """Load an .h5ad file and return its .X attribute and metadata as DataFrames."""
    adata = ad.read_h5ad(filepath)
    data_df = pd.DataFrame(adata.X, columns=adata.var_names)
    metadata_df = pd.DataFrame(adata.obs)
    return data_df, metadata_df

ab_df, ab_metadata = load_h5ad_to_df(dir + '/AB.h5ad')
lectin_df, lectin_metadata = load_h5ad_to_df(dir + '/lectin.h5ad')

print("AB shape: ", ab_df.shape)
print("AB metadata shape: ", ab_metadata.shape)
print('AB metadata columns: ', ab_metadata.columns)

def plot_feature_distributions(df, metadata, feature_names, title, outdir):
    nrows = len(feature_names) * 2  # Each feature gets two rows
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 5 * nrows))

    for i, feature in enumerate(feature_names):
        feature_data = df.iloc[:, i].copy()
        feature_data = pd.to_numeric(feature_data, errors='coerce')  # Ensure data is numeric

        # Reset indices to ensure alignment
        feature_data.reset_index(drop=True, inplace=True)
        metadata.reset_index(drop=True, inplace=True)
        
        combined_data = pd.DataFrame({
            'Counts': feature_data,
            'Celltype Major': metadata['celltype_major']
        })

        # Histogram/Density plot
        sns.histplot(feature_data.dropna(), ax=axes[2 * i], bins=30, kde=True)
        axes[2 * i].set_title(f'Distribution of {feature}')
        axes[2 * i].set_xlabel('Counts')
        axes[2 * i].set_ylabel('Frequency')

        # Violin plot
        if not combined_data['Counts'].dropna().empty and not combined_data['Celltype Major'].dropna().empty:
            sns.violinplot(x='Counts', y='Celltype Major', data=combined_data, ax=axes[2 * i + 1], scale='width', orient='h', cut=0)
            axes[2 * i + 1].set_xlim(axes[2 * i].get_xlim())  # Align x-axis with histogram
        else:
            axes[2 * i + 1].text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=axes[2 * i + 1].transAxes)
            axes[2 * i + 1].set_title('No data available')

    plt.tight_layout()
    plt.savefig(f'{outdir}/{title}.png')


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
plot_feature_distributions(lectin_df, lectin_metadata, feature_names, 'Glycan features', outdir)

feature_names = ('ADT_SSEA3_IgM', 'ADT_CA19.9', 'ADT_BloodGroupAB_Ag', 'ADT_SSEA4', 'ADT_GD2', 'ADT_GD3',
                  'ADT_Globo_H', 'ADT_CD17', 'ADT_SSEA3_IgG', 'ADT_CD77', 'ADT_MUC16', 'ADT_MUC1', 'ADT_Siglec9', 
                  'ADT_Siglec8', 'ADT_CD370', 'ADT_CD207', 'ADT_CD325', 'ADT_CD144', 'ADT_CD309', 'ADT_CD62E', 'ADT_CD106', 'ADT_CD224', 
                  'ADT_EGFR', 'ADT_CD140a', 'ADT_CD140b', 'ADT_CD193', 'ADT_Notch', 'ADT_XCR1', 'ADT_CD357', 'ADT_KLRG1')


filepath = dir + '/AB.h5ad'
plot_feature_distributions(ab_df, ab_metadata, feature_names, 'Protein features', outdir)
