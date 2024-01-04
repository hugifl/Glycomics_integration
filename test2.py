from pathlib import Path
import h5py
import numpy as np

def get_common_genes(dataset_paths):
    gene_sets = []
    for path in dataset_paths:
        with h5py.File(path, 'r') as file:
            genes = list(file['exp']['rowData'])
            gene_sets.append(set(genes))
    
    common_genes = set.intersection(*gene_sets)
    return common_genes

def filter_counts(file_path, common_genes):
    with h5py.File(file_path, 'r') as file:
        all_genes = list(file['exp']['rowData'])
        gene_indices = [all_genes.index(gene) for gene in common_genes if gene in all_genes]
        counts = file['exp']['counts'][:]
        filtered_counts = counts[:, gene_indices]
        
        # Get the names of the filtered genes
        filtered_gene_names = [all_genes[i] for i in gene_indices]
        print("before filetering", all_genes[:20])
        print("indexs", gene_indices[:20])
        print("after filetering", filtered_gene_names[:20])
        
        return filtered_counts, filtered_gene_names


OUTIDR = Path('/cluster/scratch/hugifl/TitrationII')
AB_path_0_1 = OUTIDR / 'AB.0.1_mikrog.GEX_cellrangerADT_SCE.h5'
AB_path_1 = OUTIDR / 'AB.1_mikrog.GEX_cellrangerADT_SCE.h5'
dataset_paths = [AB_path_0_1, AB_path_1]

# Get the common genes
common_genes = get_common_genes(dataset_paths)

# Filter counts data in each dataset
all_filtered_counts = [filter_counts(path, common_genes) for path in dataset_paths]




#def get_gene_list(file_path):
#    with h5py.File(file_path, 'r') as file:
#        exp_group = file['exp']
#        row_data = exp_group['rowData']
#        return list(row_data)
#
#
## Extract gene lists from each dataset
#gene_lists_0_1 = get_gene_list(AB_path_0_1) 
#gene_lists_1 = get_gene_list(AB_path_1) 
#
#print("Number of genes in AB.0.1_mikrog.GEX_cellrangerADT_SCE.h5:", len(gene_lists_0_1))
#print("Number of genes in AB.1_mikrog.GEX_cellrangerADT_SCE.h5:", len(gene_lists_1))
#print("first 20 genes in AB.0.1_mikrog.GEX_cellrangerADT_SCE.h5:", gene_lists_0_1[:20])
#print("first 20 genes in AB.1_mikrog.GEX_cellrangerADT_SCE.h5:", gene_lists_1[:20])