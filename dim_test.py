import anndata

datasets = ['1_glycomics', '2_glycomics','3_glycomics','4_glycomics','5_glycomics','6_glycomics','9_glycomics'] 


for dataset in datasets:
    lectin_path = '/cluster/scratch/hugifl/' + dataset + '/lectin.h5ad'
    lectin_data = anndata.read(lectin_path)
    AB_path = '/cluster/scratch/hugifl/' + dataset + '/AB.h5ad'
    AB_data = anndata.read(AB_path)
    print(dataset)
    # dimensions of lectin observation matrix
    print("dimensions of lectin observation matrix: ", lectin_data.X.shape)
    # dimensions of AB observation matrix
    print("dimensions of AB observation matrix: ", AB_data.X.shape)
    
