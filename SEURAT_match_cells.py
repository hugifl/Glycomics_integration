# This script is used to apply the same cell matching method to the integrated AB and lectin dataset from SEURAT that was used in the SCIM paper.

import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
from anndata import AnnData
import scanpy as sc
import csv
import numpy as np
from scim.matching import get_cost_knn_graph, mcmf

def extract_matched_labels(celltype_major_AB, celltype_major_lectin, row_idx, col_idx):
    """
    Merge cell type labels for matched cells from two technologies.
    celltype_major_AB, celltype_major_lectin: Series with celltype labels for each technology.
    row_idx, col_idx: Numerical indices of the matches (AB and lectin, respectively).
    """
    # Ensure row_idx and col_idx are of the same length
    indices = pd.DataFrame({'AB_index': row_idx, 'lectin_index': col_idx}).dropna()
    indices['AB_index'] = indices['AB_index'].astype(int)
    indices['lectin_index'] = indices['lectin_index'].astype(int)

    # Extract matched labels
    matched_labels_AB = celltype_major_AB.iloc[indices['AB_index'].values].reset_index(drop=True)
    matched_labels_lectin = celltype_major_lectin.iloc[indices['lectin_index'].values].reset_index(drop=True)

    # Create a DataFrame from the matched labels
    labels_matched = pd.DataFrame({
        'celltype_AB': matched_labels_AB,
        'celltype_lectin': matched_labels_lectin
    })

    return labels_matched

def get_accuracy(matches, colname_compare='celltype_', tech1 = 'AB', tech2 = "lectin", n_null=0, extended=True):
    """Compute accuracy as true positive fraction
    matches: pandas dataframe, output from extract_matched_labels()
    colname_compare: column name to use for accuracy calculation {colname_compare}_source,
                     {colname_compare}_target must be in matches.columns
    n_null: number of matches with the null node to account for in denominator
    extended: whether to return extended information {accuracy, n_tp, n_fp}
    """
    n_tp = np.sum(matches[colname_compare+tech1]==matches[colname_compare+tech2])
    n_matches = matches.shape[0] + n_null
    accuracy = n_tp/n_matches
    if(extended):
        return accuracy, n_tp, n_matches-n_tp
    else:
        return accuracy

def get_confusion_matrix(matches, colname_compare='celltype_', tech1 = 'AB', tech2 = "lectin"):
    """ Get the confusion matrix
    matches: pandas dataframe, output from extract_matched_labels()
    colname_compare: column name to use for accuracy calculation {colname_compare}_source,
                     {colname_compare}_target must be in matches.columns
    """
    if(np.sum(matches.columns.isin([colname_compare+tech1, colname_compare+tech2]))!=2):
        print('The input dataframe does not include {colname_compare} information!')
    else:
        y_source = pd.Series(matches[colname_compare+tech1], name=tech1)
        y_target = pd.Series(matches[colname_compare+tech2], name=tech2)
    
        return pd.crosstab(y_source, y_target)
    
# Paths
outdir = '/cluster/home/hugifl/scim/plots_R/'
datadir = '/cluster/home/hugifl/scim/seurat_data_2/'
dataset = 'rna'

index_lectin = 4301
PC_dimensions = 29

# Load data
pca_embeddings = pd.read_csv(f"{datadir}{dataset}_PCA_coordinates.csv", index_col=0)
celltype_major = pd.read_csv(f"{datadir}{dataset}_celltype_major.csv")['x'].astype(str)  # Load 'x' column as string
celltype_final = pd.read_csv(f"{datadir}{dataset}_celltype_final.csv")['x'].astype(str)  # Load 'x' column as string

max_PC_dimensions = pca_embeddings.shape[1]
if PC_dimensions > max_PC_dimensions:
    print(f"Error: PC_dimensions must be less than or equal to {max_PC_dimensions}")
    exit()
else:
    pca_embeddings = pca_embeddings.iloc[:,:PC_dimensions]

# Split the DataFrame
AB_PCA_embeddings = pca_embeddings.iloc[:index_lectin]
celltype_major_AB = celltype_major.iloc[:index_lectin]
lectin_PCA_embeddings = pca_embeddings.iloc[index_lectin:]
celltype_major_lectin = celltype_major.iloc[index_lectin:]

# Optionally, reset index for the new DataFrames
AB_PCA_embeddings = AB_PCA_embeddings.reset_index(drop=True)
lectin_PCA_embeddings = lectin_PCA_embeddings.reset_index(drop=True)
celltype_major_AB = celltype_major_AB.reset_index(drop=True)
celltype_major_lectin = celltype_major_lectin.reset_index(drop=True)

# Rename the index for clarity
lectin_PCA_embeddings.index = ['lectin_' + str(i) for i in range(len(lectin_PCA_embeddings))]
AB_PCA_embeddings.index = ['AB_' + str(i) for i in range(len(AB_PCA_embeddings))]
celltype_major_lectin.index = ['lectin_' + str(i) for i in range(len(celltype_major_lectin))]
celltype_major_AB.index = ['AB_' + str(i) for i in range(len(celltype_major_AB))]

# Print the head of the new DataFrames to verify
print(AB_PCA_embeddings.head())
print(celltype_major_lectin.head())

G = get_cost_knn_graph(AB_PCA_embeddings, lectin_PCA_embeddings, knn_k=20, null_cost_percentile=95, capacity_method='uniform')

# Run mcmf and extract matches
row_ind, col_ind = mcmf(G)
matches = extract_matched_labels(celltype_major_AB, celltype_major_lectin, row_ind, col_ind)
matches.to_csv(f"{datadir}{dataset}_matches_{PC_dimensions}.csv", index=False)

    
accuracy, n_tp, n_matches_min_n_tp = get_accuracy(matches, colname_compare='celltype_')

accuracy_file = f'{dataset}_celltype_accuracy_{PC_dimensions}.csv'
with open(datadir + accuracy_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow([accuracy])

print('accuracy (celltype_major): ', accuracy)
confusion_matrix = get_confusion_matrix(matches, colname_compare='celltype_')
print('confusion matrix (celltype_major): ', confusion_matrix)
#save confusion matrix
confusion_matrix.to_csv(f"{datadir}{dataset}_confusion_matrix_{PC_dimensions}.csv", index=False)
