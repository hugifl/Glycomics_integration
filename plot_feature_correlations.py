import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np


model = '9_glycomics_6_2'
file_path =  '/cluster/scratch/hugifl/9_glycomics_6_2/integrated_features_and_labels.csv' #f'/cluster/scratch/hugifl/{model}/integrated_features_and_labels.csv'
df = pd.read_csv(file_path)
outpath = f'/cluster/home/hugifl/scim/plots/feature_correlations/{model}'
RNA_features = 0
correlation_method = 'spearman'

lectin_features = ['ADT_AAL', 'ADT_PNA', 'ADT_UEA.II', 'ADT_LcH', 'ADT_PHA_E', 'ADT_SNA', 'ADT_VVA', 'ADT_PSA', 'ADT_ECA', 'ADT_HPA', 'ADT_Jacalin', 'ADT_RCA', 'ADT_WGA', 'ADT_UEA.I', 'ADT_ConA', 'ADT_AOL']
AB_features = ['ADT_SSEA3_IgM', 'ADT_CA19.9', 'ADT_BloodGroupAB_Ag', 'ADT_SSEA4', 'ADT_GD2', 'ADT_GD3', 'ADT_Globo_H', 'ADT_CD17', 'ADT_SSEA3_IgG', 'ADT_CD77', 'ADT_MUC16', 'ADT_MUC1', 'ADT_Siglec9', 'ADT_Siglec8', 'ADT_CD370', 'ADT_CD207', 'ADT_CD325', 'ADT_CD144', 'ADT_CD309', 'ADT_CD62E', 'ADT_CD106', 'ADT_CD224', 'ADT_EGFR', 'ADT_CD140a', 'ADT_CD140b', 'ADT_CD193', 'ADT_Notch', 'ADT_XCR1', 'ADT_CD357', 'ADT_KLRG1']

lectin_features = [feature[4:] for feature in lectin_features]
AB_features = [feature[4:] for feature in AB_features]


if not os.path.exists(outpath):
    os.makedirs(outpath)

def reorder_columns(df):
    # Helper function to extract the numerical part from the column name
    def extract_number(col_name):
        return int(col_name.split('_')[-1])
    
    # Extract column names for lectin and AB features
    lectin_cols = [col for col in df.columns if col.startswith('lectin_feature_')]
    AB_cols = [col for col in df.columns if col.startswith('AB_feature_')]
    
    # Sort the columns based on their numerical part
    lectin_cols_sorted = sorted(lectin_cols, key=extract_number)
    AB_cols_sorted = sorted(AB_cols, key=extract_number)
    
    # Check if lectin features already come before AB features (based on the first occurrence in the DataFrame)
    if df.columns.tolist().index(lectin_cols[0]) > df.columns.tolist().index(AB_cols[0]):
        # If AB features come first, reorder DataFrame columns
        new_order = ['tech'] + lectin_cols_sorted + AB_cols_sorted
        df = df[new_order]
    else:
        # If lectin features already come first, just ensure they're sorted correctly
        new_order = ['tech'] + lectin_cols_sorted + AB_cols_sorted
        df = df[new_order]
    
    return df

def calculate_and_plot_correlations(df, df_name, correlation_method, outpath, RNA_features):
    outpath = f'{outpath}/{df_name}'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    print("shape of df is", df.shape)
    print("head of df is", df.head())
    print("colnames of df are", df.columns)
    df_lectin = df[df['tech'] == 'lectin']
    df_AB = df[df['tech'] == 'AB']
    print("RNA_features is", RNA_features)
    
    lectin_feature_cols = [col for col in df.columns if 'lectin_feature_' in col]
    AB_feature_cols = [col for col in df.columns if 'AB_feature_' in col]
    total_lectin_features = len(lectin_feature_cols)
    total_AB_features = len(AB_feature_cols)
    total_lectin_ADT_features = total_lectin_features - RNA_features
    total_AB_ADT_features = total_AB_features - RNA_features
    print("total lectin ADT features is", total_lectin_ADT_features)
    print("total AB ADT features is", total_AB_ADT_features)

    for tech, tech_df in zip(['lectin', 'AB', 'full'], [df_lectin, df_AB, df]):
        rna_lectin_features = tech_df.iloc[:, 1+total_lectin_ADT_features:1+total_lectin_features]
        non_rna_lectin_features = tech_df.iloc[:, 1:1+total_lectin_ADT_features]
        rna_ab_features = tech_df.iloc[:, 1+total_lectin_features+total_AB_ADT_features:]
        non_rna_ab_features = tech_df.iloc[:, 1+total_lectin_features: 1+total_lectin_features+total_AB_ADT_features]

        non_rna_correlation_df = pd.DataFrame(index=non_rna_lectin_features.columns, columns=non_rna_ab_features.columns)
        for lectin_col in non_rna_lectin_features.columns:
            for ab_col in non_rna_ab_features.columns:
                correlation = non_rna_lectin_features[lectin_col].corr(non_rna_ab_features[ab_col], method=correlation_method)
                non_rna_correlation_df.at[lectin_col, ab_col] = correlation
        print("shape non_rna_correlation_df is", non_rna_correlation_df.shape)
        non_rna_correlation_df.index = lectin_features
        non_rna_correlation_df.columns = AB_features

        if RNA_features > 0:
            correlations = []
            rna_correlation_df = pd.DataFrame(index=rna_lectin_features.columns, columns=rna_ab_features.columns)
            for lectin_col in rna_lectin_features.columns:
                for ab_col in rna_ab_features.columns:
                    correlation = rna_lectin_features[lectin_col].corr(rna_ab_features[ab_col], method=correlation_method)
                    rna_correlation_df.at[lectin_col, ab_col] = correlation

            for lectin_col, ab_col in zip(rna_lectin_features.columns, rna_ab_features.columns):
                correlation = rna_lectin_features[lectin_col].corr(rna_ab_features[ab_col], method=correlation_method)
                correlations.append(correlation)
                average_correlation = np.nanmean(correlations)
            print(f'{df_name} Average self correlation between RNA features: {average_correlation} ({tech})')
           
            with open(outpath + f'/{df_name}_{tech}_RNA_{correlation_method}_average_RNA_self_correlation.txt', 'w') as f:
                f.write(f'{average_correlation}')

        plt.figure(figsize=(40, 22))
        ax = sns.heatmap(non_rna_correlation_df.astype(float), annot=False, cmap='coolwarm', yticklabels=lectin_features, xticklabels=AB_features)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=24)    
        plt.title('Correlation Heatmap Between Surface Glycans and Surface Proteins', fontsize=28)
        plt.xlabel('Surface Proteins', fontsize=24)
        plt.ylabel('Surface Glycans', fontsize=24)
        plt.xticks(fontsize=20)  
        plt.yticks(fontsize=20)   
        plt.savefig(outpath + f'/{df_name}_{tech}_ADT_{correlation_method}_correlations.png', dpi=300)


        if RNA_features > 0: 
            plt.figure(figsize=(30, 20))
            ax = sns.heatmap(rna_correlation_df.astype(float), annot=False, cmap='coolwarm')
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=24) 
            plt.title('Control Correlation scRNA Features', fontsize=28)
            plt.xlabel('AB RNA Features', fontsize=24)
            plt.ylabel('Lectin RNA Features', fontsize=24)
            plt.xticks(fontsize=14)  
            plt.yticks(fontsize=14)
            plt.savefig(outpath + f'/{df_name}_{tech}_RNA_{correlation_method}_correlations.png', dpi=300)

celltypes = df['original_label'].unique()
for celltype in celltypes:
    celltype_df = df[df['original_label'] == celltype]

    celltype_df = celltype_df.iloc[:, :-2]
    celltype_df = reorder_columns(celltype_df)
    
    celltype_name = f'celltype_{str(celltype)}'
    calculate_and_plot_correlations(celltype_df, celltype_name, correlation_method, outpath, RNA_features)

    
df = df.iloc[:, :-2]
df = reorder_columns(df)
calculate_and_plot_correlations(df, 'celltypes_pooled', correlation_method, outpath, RNA_features)



