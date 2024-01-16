from pathlib import Path
import h5py
import numpy as np


path = '/cluster/scratch/hugifl/glycomics_4_less_KL_more_units_in_lat/AB.0.1_mikrog.GEX_cellrangerADT_SCE.h5'

with h5py.File(path, 'r') as file:
    row_data = file['exp']['rowData']
    col_data = file['exp']['colData']
    print("col data variable names: ", col_data.dtype.names)