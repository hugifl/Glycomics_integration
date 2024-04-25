
import os
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy

def remove_specified_columns(df):
    # Filter out columns based on your conditions
    filtered_columns = [col for col in df.columns if not (
        (col.startswith('AB_feature_') and int(col.split('_')[-1]) > 30) or
        (col.startswith('lectin_feature_') and int(col.split('_')[-1]) > 16)
    )]
    return df[filtered_columns]


# Define the feature names
lectin_feature_names =  ['ADT_AAL', 'ADT_PNA', 'ADT_UEA.II', 'ADT_LcH', 'ADT_PHA_E', 'ADT_SNA', 
    'ADT_VVA', 'ADT_PSA', 'ADT_ECA', 'ADT_HPA', 'ADT_Jacalin', 'ADT_RCA', 
    'ADT_WGA', 'ADT_UEA.I', 'ADT_ConA', 'ADT_AOL']


ab_feature_names = ['ADT_SSEA3_IgM', 'ADT_CA19.9', 'ADT_BloodGroupAB_Ag', 'ADT_SSEA4', 'ADT_GD2', 'ADT_GD3',
                  'ADT_Globo_H', 'ADT_CD17', 'ADT_SSEA3_IgG', 'ADT_CD77', 'ADT_MUC16', 'ADT_MUC1', 'ADT_Siglec9', 
                  'ADT_Siglec8', 'ADT_CD370', 'ADT_CD207', 'ADT_CD325', 'ADT_CD144', 'ADT_CD309', 'ADT_CD62E', 'ADT_CD106', 'ADT_CD224', 
                  'ADT_EGFR', 'ADT_CD140a', 'ADT_CD140b', 'ADT_CD193', 'ADT_Notch', 'ADT_XCR1', 'ADT_CD357', 'ADT_KLRG1']

RNA = True
dataset = 'top_3_all_RNA'
# Load data from CSV
outdir = f'/cluster/home/hugifl/scim/feature_count_distributions_per_celltype_vs_imputed_{dataset}'
if not os.path.exists(outdir):
    os.makedirs(outdir)
data = pd.read_csv('/cluster/scratch/hugifl/4_glycomics_7_2/integrated_features_and_labels.csv')


if RNA:
    data = remove_specified_columns(data)

# Extract columns for real and imputed data
ab_columns = [col for col in data.columns if 'AB_feature_' in col]
lectin_columns = [col for col in data.columns if 'lectin_feature_' in col]

# Create DataFrames for real and imputed data with appropriate metadata
df_real_AB = data[data['tech'] == 'AB'][['original_label'] + ab_columns]
df_imputed_lectin = data[data['tech'] == 'AB'][['original_label'] + lectin_columns]
df_real_lectin = data[data['tech'] == 'lectin'][['original_label'] + lectin_columns]
df_imputed_AB = data[data['tech'] == 'lectin'][['original_label'] + ab_columns]

# Rename columns to use actual feature names for clarity in plotting
df_real_AB.columns = ['original_label'] + ab_feature_names
df_imputed_lectin.columns = ['original_label'] + lectin_feature_names
df_real_lectin.columns = ['original_label'] + lectin_feature_names
df_imputed_AB.columns = ['original_label'] + ab_feature_names

print("columns for real AB data: ", df_real_AB.columns)


def plot_feature_distributions1(df, metadata_col, feature_names, technology, outdir):
    # Calculate the number of rows needed: each feature gets two rows (one for histogram, one for violin),
    # and we are displaying up to four features per row.
    ncols = 4  # Number of features per row
    nrows = ((len(feature_names) + ncols - 1) // ncols) * 2  # Two rows (histogram + violin) per feature set

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))  # Create a grid of subplots

    # Handle fewer axes than needed for reshaping when fewer than 'ncols' features
    axes = axes.reshape(nrows, ncols)

    for i, feature in enumerate(feature_names):
        col_index = i % ncols  # Column position in the grid
        row_base = (i // ncols) * 2  # Starting row for this feature set (0, 2, 4, ...)

        # Convert data to numeric and handle NaNs
        feature_data = pd.to_numeric(df[feature], errors='coerce').dropna()
        combined_data = pd.DataFrame({
            'Counts': feature_data,
            'Celltype Major': df[metadata_col]
        }).dropna()

        # Histogram/Density plot
        ax_hist = axes[row_base, col_index]
        sns.histplot(feature_data, ax=ax_hist, bins=30, kde=True)
        ax_hist.set_title(f'Real {technology} Distribution of {feature}')
        ax_hist.set_xlabel('Counts')
        ax_hist.set_ylabel('Frequency')

        # Violin plot
        ax_violin = axes[row_base + 1, col_index]
        if not combined_data.empty:
            sns.violinplot(x='Counts', y='Celltype Major', data=combined_data, ax=ax_violin, scale='width', orient='h', cut=0)
            ax_violin.set_xlim(ax_hist.get_xlim())  # Align x-axis with histogram
        else:
            ax_violin.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=ax_violin.transAxes)
            ax_violin.set_title('No data available')

    plt.tight_layout()
    plt.savefig(f'{outdir}/{technology}_features.png')

def plot_feature_distributions(df_real, df_imputed, metadata_col, feature_names, technology, outdir):
    ncols = 4  # Number of features per row
    nrows = ((len(feature_names) + ncols - 1) // ncols) * 2  # Two rows per feature set

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
    axes = axes.reshape(nrows, ncols)  # Reshape axes to ensure proper indexing

    for i, feature in enumerate(feature_names):
        col_index = i % ncols
        row_base = (i // ncols) * 2

        # Convert data to numeric and handle NaNs
        real_data = pd.to_numeric(df_real[feature], errors='coerce').dropna()
        imputed_data = pd.to_numeric(df_imputed[feature], errors='coerce').dropna()

        # Density plot
        ax_kde = axes[row_base, col_index]
        sns.kdeplot(real_data, ax=ax_kde, color='blue', alpha=0.5, label='Real')
        sns.kdeplot(imputed_data, ax=ax_kde, color='red', alpha=0.5, label='Imputed')
        ax_kde.set_title(f'{technology} Distribution of {feature}')
        ax_kde.set_xlabel('Counts')
        ax_kde.set_ylabel('Density')
        ax_kde.legend()

        # Prepare combined data for violin plots, ensure 'Celltype Major' is categorical
        combined_real_data = pd.DataFrame({
            'Counts': real_data,
            'Celltype Major': pd.Categorical(df_real[metadata_col]),
            'Type': 'Real'
        })
        combined_imputed_data = pd.DataFrame({
            'Counts': imputed_data,
            'Celltype Major': pd.Categorical(df_imputed[metadata_col]),
            'Type': 'Imputed'
        })
        combined_data = pd.concat([combined_real_data, combined_imputed_data])

        # Violin plot
        ax_violin = axes[row_base + 1, col_index]
        if not combined_data.empty:
            sns.violinplot(x='Counts', y='Celltype Major', hue='Type', data=combined_data, ax=ax_violin, split=True, palette={'Real': 'blue', 'Imputed': 'red'}, alpha=0.7)
        else:
            ax_violin.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=ax_violin.transAxes)
            ax_violin.set_title('No data available')

    plt.tight_layout()
    plt.savefig(f'{outdir}/{technology}_features_combined.png')

plot_feature_distributions(df_real_AB, df_imputed_AB, 'original_label', ab_feature_names, 'AB', outdir)
plot_feature_distributions(df_real_lectin, df_imputed_lectin, 'original_label', lectin_feature_names, 'Lectin', outdir)


#plot_feature_distributions(df_real_AB, 'original_label', ab_feature_names, 'AB', outdir)
#plot_feature_distributions(df_real_lectin, 'original_label', lectin_feature_names, 'Lectin', outdir)
#plot_feature_distributions(df_imputed_AB, 'original_label', ab_feature_names, 'AB_Imputed', outdir)
#plot_feature_distributions(df_imputed_lectin, 'original_label', lectin_feature_names, 'Lectin_Imputed', outdir)