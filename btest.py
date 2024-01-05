import h5py
from pathlib import Path
import numpy as np

def print_structure(name, obj):
    """Recursively print the structure of the HDF5 file."""
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name} - Shape: {obj.shape}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

def print_dataset_info(group, dataset_name):
    """Prints information about a dataset within an HDF5 group."""
    if dataset_name in group:
        dataset = group[dataset_name]
        print(f"Dataset '{dataset_name}' dimensions: {dataset.shape}")
        print(f"Example entries from '{dataset_name}':\n{dataset[0:min(5, dataset.shape[0])]}")
    else:
        print(f"Dataset '{dataset_name}' not found in the group.")

def compare_genes(group, dataset_name):
    """Prints information about a dataset within an HDF5 group."""
    if dataset_name in group:
        dataset = group[dataset_name]
        print(f"Dataset '{dataset_name}' dimensions: {dataset.shape}")
        print(f"Example entries from '{dataset_name}':\n{dataset[0:min(20, dataset.shape[0])]}")
    else:
        print(f"Dataset '{dataset_name}' not found in the group.")

def compare_celltypes(group, dataset_name):
    """Prints information about a dataset within an HDF5 group."""
    if dataset_name in group:
        dataset = group[dataset_name]
        print(f"Dataset '{dataset_name}' dimensions: {dataset.shape}")
        celltypes_final = dataset['celltype_major_full_ct_name']
        celltypes_major = dataset['celltype_major_full_ct_name']
        unique_celltypes_final = np.unique(celltypes_final)
        unique_celltypes_major = np.unique(celltypes_major)
        print(f"Unique celltypes final '{dataset_name}': ",unique_celltypes_final)
        print(f"Unique celltypes major '{dataset_name}': ",unique_celltypes_major)
    else:
        print(f"Dataset '{dataset_name}' not found in the group.")

def main(hdf5_path, viable_conc):
    OUTDIR = Path('/cluster/scratch/hugifl/TitrationII')
    dataset_types = ['AB', 'lectin']
    with h5py.File(hdf5_path, 'r') as file:
        print(f"Structure of HDF5 file: {hdf5_path}")
        file.visititems(print_structure)
    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as file:
        print(f"Opened HDF5 file: {hdf5_path}")

        # Access the 'exp' group
      
        exp_group = file['exp']

        # Print information about datasets inside 'exp' group
        #print_dataset_info(exp_group, 'colData')
        #print_dataset_info(exp_group, 'counts')
        #print_dataset_info(exp_group, 'rowData')

    for dataset_type in dataset_types:
        for conc in viable_conc:
            print(f"Dataset type: {dataset_type}, concentration: {conc}")
            path = OUTDIR / f'{dataset_type}.{conc}_mikrog.GEX_cellrangerADT_SCE.h5'
            with h5py.File(path, 'r') as file:
                exp_group = file['exp']
                print("dimensions of counts: ", exp_group['counts'].shape)
                #compare_genes(exp_group, 'rowData')
                #compare_celltypes(exp_group, 'colData')
                colData = exp_group['colData']
                print(colData.dtype.names)
       
viable_conc = [0.1,  0.5, 1, 2]
# Replace 'your_file.h5' with the path to your HDF5 file
hdf5_path = '/cluster/scratch/hugifl/TitrationII/AB.0.1_mikrog.GEX_cellrangerADT_SCE.h5'
main(hdf5_path, viable_conc)