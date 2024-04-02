
import h5py
from pathlib import Path
import numpy as np

path = Path('/cluster/scratch/hugifl/4_glycomics_7_2_c2/AB.2_mikrog.GEX_cellrangerADT_SCE.h5')

with h5py.File(path, 'r') as file:
    exp_group = file['exp']
    col_data = exp_group['colData']
    row_data = exp_group['rowData']
    adt_fields = [field for field in col_data.dtype.names if field.startswith('ADT_') and 'Hashtag' not in field and 'barcodes' not in field]
    print("strucutre of the file", col_data.dtype.names)
    print("celltype_major", col_data['celltype_major'])
    celltype_major = col_data['celltype_major'][:]
    unique_celltypes, counts = np.unique(celltype_major, return_counts=True)
    
    # Print unique cell types and their abundance
    for cell_type, count in zip(unique_celltypes, counts):
        # Assuming no decoding is needed; remove the decode call
        print(cell_type, count)
    