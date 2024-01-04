from pathlib import Path
import h5py
import numpy as np


genes_1 = ["A", "B", "C", "F", "G", "H", "I", "K"]
genes_2 = ["A", "B", "C", "F"]
gene_list = [genes_1, genes_2]
counts_1 = [1,2,3,4,5,6,7,8]
counts_2 = [5,6,7,8]
count_list = [counts_1, counts_2]


def get_common_genes(dataset_paths):
    gene_sets = []
    for list in dataset_paths:
        gene_sets.append(set(list))
    
    common_genes = set.intersection(*gene_sets)
    return common_genes

def filter_counts(all_genes, counts, common_genes):
    gene_indices = [all_genes.index(gene) for gene in common_genes if gene in all_genes]
    filtered_counts = [counts[i] for i in gene_indices]  # Modified line
    
    # Get the names of the filtered genes
    filtered_gene_names = [all_genes[i] for i in gene_indices]
    return filtered_counts, filtered_gene_names


# Get the common genes
common_genes = get_common_genes(gene_list)

# Filter counts data in each dataset
filtered_counts_1, filtered_gene_names_1 = filter_counts(genes_1, counts_1, common_genes)
filtered_counts_2, filtered_gene_names_2 = filter_counts(genes_2, counts_2, common_genes)


print("filtered_gene_names_1", filtered_gene_names_1)
print("filtered_counts_1", filtered_counts_1)
print("filtered_gene_names_2", filtered_gene_names_2)
print("filtered_counts_2", filtered_counts_2)