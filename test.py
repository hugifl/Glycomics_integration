
import h5py
from pathlib import Path


OUTIDR = Path('/cluster/scratch/hugifl/9_glycomics_6_2')
dataset_type='lectin'
AB_viable_conc = [0.1, 0.25, 0.5, 1, 2]
lectin_viable_conc = [0.1, 0.5, 1, 2] # there is 0.0 
adt_fields_collection = []
for conc in lectin_viable_conc:
    path = OUTIDR / f'{dataset_type}.{conc}_mikrog.GEX_cellrangerADT_SCE_common_genes.h5'
    with h5py.File(path, 'r') as file:
        exp_group = file['exp']
        col_data = exp_group['colData']
        adt_fields = [field for field in col_data.dtype.names if field.startswith('ADT_') and 'Hashtag' not in field and 'barcodes' not in field]
        adt_fields_collection.append(adt_fields)
        print('lecin',len(adt_fields))

#check if all adt_fields are the same
for i in range(1, len(adt_fields_collection)):
    if adt_fields_collection[i] != adt_fields_collection[i-1]:
        print('not the same')
    else:
        print('same')


print(adt_fields_collection[0])
dataset_type='AB'
adt_fields_collection = []
for conc in AB_viable_conc:
    path = OUTIDR / f'{dataset_type}.{conc}_mikrog.GEX_cellrangerADT_SCE_common_genes.h5'
    with h5py.File(path, 'r') as file:
        exp_group = file['exp']
        col_data = exp_group['colData']
        adt_fields = [field for field in col_data.dtype.names if field.startswith('ADT_') and 'Hashtag' not in field and 'barcodes' not in field]
        adt_fields_collection.append(adt_fields)
        print('AB',len(adt_fields))

#check if all adt_fields are the same
for i in range(1, len(adt_fields_collection)):
    if adt_fields_collection[i] != adt_fields_collection[i-1]:
        print('not the same')
    else:
        print('same')


print(adt_fields_collection[0])