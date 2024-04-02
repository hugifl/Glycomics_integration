import os
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy

normalization_methods = ['CLR_cell_nolog','CLR_feature_nolog','library'] 
technology_files = ['AB.h5ad', 'lectin.h5ad']
data_dirs = []
for normalization_method in normalization_methods:
    data_dirs.append(f'/cluster/scratch/hugifl/9_glycomics_6_3_c2_{normalization_method}')

outdir = f'/cluster/home/hugifl/scim/normalization_methods_feature_count_correlations'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def load_h5ad_to_df(filepath):
    """Load an .h5ad file and return its .X attribute as a DataFrame."""
    adata = ad.read_h5ad(filepath)
    # Convert the .X (data matrix) to a DataFrame, assuming it's stored as an ndarray or similar
    # For sparse matrices, you might need to convert them to dense: .toarray() or .todense()
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    return df

def plot_feature_distributions(data_dirs, technology_files, normalization_methods, outdir):
    for technology_file in technology_files:
        data_frames = []
        for data_dir in data_dirs:
            filepath = os.path.join(data_dir, technology_file)
            df = load_h5ad_to_df(filepath)
            data_frames.append(df)

        for i in range(len(normalization_methods)):
            for j in range(i + 1, len(normalization_methods)):
                method1, method2 = normalization_methods[i], normalization_methods[j]
                df1, df2 = data_frames[i], data_frames[j]

                correlations = []
                for feature in df1.columns:
                    corr = scipy.stats.pearsonr(df1[feature], df2[feature])[0]
                    correlations.append(corr)

                n_features = len(df1.columns)
                n_rows = np.ceil(n_features / 5).astype(int)
                fig, axs = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows), constrained_layout=True)
                fig.suptitle(f'Correlations between {method1} and {method2} for {technology_file.split(".")[0]} features')

                if n_features < 5:
                    axs = axs.flatten()
                    for ax in axs[n_features:]:
                        ax.set_visible(False)

                for idx, (corr, feature) in enumerate(zip(correlations, df1.columns)):
                    if n_features >= 5:
                        ax = axs[idx // 5, idx % 5]
                    else:
                        ax = axs[idx]
                    sns.histplot(df1[feature], ax=ax, color='blue', alpha=0.5, label=method1)
                    sns.histplot(df2[feature], ax=ax, color='red', alpha=0.5, label=method2)
                    ax.set_title(f'{feature}\nCorr: {corr:.2f}')
                    ax.legend()

                plt.savefig(os.path.join(outdir, f'feature_counts_{technology_file.split(".")[0]}_{method1}_vs_{method2}_correlations.png'))
                plt.close()

def plot_feature_scatter_correlations(data_dirs, technology_files, normalization_methods, outdir):
    for technology_file in technology_files:
        data_frames = []
        # Load data for each normalization method
        for data_dir in data_dirs:
            filepath = os.path.join(data_dir, technology_file)
            df = load_h5ad_to_df(filepath)
            data_frames.append(df)

        # Compare each pair of normalization methods
        for i in range(len(normalization_methods)):
            for j in range(i + 1, len(normalization_methods)):
                method1, method2 = normalization_methods[i], normalization_methods[j]
                df1, df2 = data_frames[i], data_frames[j]

                # Determine number of features and setup plot grid
                n_features = len(df1.columns)
                n_rows = np.ceil(n_features / 5).astype(int)
                fig, axs = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows), constrained_layout=True)
                fig.suptitle(f'Scatter Correlations between {method1} and {method2} for {technology_file.split(".")[0]}')

                # Flatten axs array for easy indexing if it's multidimensional
                if n_features >= 5:
                    axs = axs.flatten()

                # Hide unused subplots if features < 5
                for ax in axs[n_features:]:
                    ax.set_visible(False)

                # Plot scatter correlations for each feature
                for idx, feature in enumerate(df1.columns):
                    ax = axs[idx]
                    ax.scatter(df1[feature], df2[feature], alpha=0.5)
                    ax.set_title(f'{feature}')
                    ax.set_xlabel(method1)
                    ax.set_ylabel(method2)
                    # Calculate and display Pearson correlation coefficient
                    corr_coef = scipy.stats.pearsonr(df1[feature], df2[feature])[0]
                    ax.text(0.05, 0.95, f'Corr: {corr_coef:.2f}', transform=ax.transAxes, verticalalignment='top')

                # Save the plot
                plt.savefig(os.path.join(outdir, f'{technology_file.split(".")[0]}_{method1}_vs_{method2}_scatter_correlations.png'))
                plt.close()

plot_feature_distributions(data_dirs, technology_files, normalization_methods, outdir)
plot_feature_scatter_correlations(data_dirs, technology_files, normalization_methods, outdir)
