# Script used to integrate Proteomics (+ scRNA) and Glycomics (+ scRNA) data.

import tensorflow as tf
import numpy as np
import pandas as pd
import anndata
import csv
from itertools import cycle
import scanpy as sc
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scim.utils import make_network
from scim.model import VAE, Integrator
from scim.discriminator import SpectralNormCritic
from scim.trainer import Trainer
from scim.utils import switch_obsm, make_network, plot_training, adata_to_pd
from scim.simulate import simulate
from scim.evaluate import score_divergence, extract_matched_labels, get_accuracy, get_confusion_matrix, get_correlation
from scim.matching import get_cost_knn_graph, mcmf
from utils_training import (get_number_of_genes, load_and_split_data, perform_pca_and_plot, 
                            perform_pca_and_plot_reconstructions, visualize_latent_space, visualize_discriminator_performance, 
                            get_label_categories, setup_models, plot_integrated_latent_space, initialize_models_and_training, 
                            plot_training_curve, integrate_target_to_source, initialize_latent_space, cache_model_outputs,
                            match_cells, evaluate_latent_space, plot_integrated_latent_space_full_data, match_cells2, load_full_data,
                            load_full_data_and_label_mapping)

# Enable eager execution for TensorFlow
tf.compat.v1.enable_eager_execution()

# Define constants and paths
OUTDIR = Path('/cluster/scratch/hugifl/4_glycomics_7_3_c2')  
TECHS = ['lectin', 'AB']
first_source = 'AB'
LABEL = 'celltype_major'
LABEL_2 = 'celltype_final'

iteration_to_use = 6
source = 'AB'
target = 'lectin'

ckpt_dir = OUTDIR / f'2_integration_iteration_{iteration_to_use}_{target}' / 'model'

def print_dataset_info(full_dataset):
    for tech, dataset in full_dataset.items():
        total_cells = 0
        unique_labels = set()

        # Iterate through each batch in the dataset
        for (data, labels, _) in dataset:
            # Accumulate total number of cells
            total_cells += data.shape[0]
            
            # Update the set of unique labels
            unique_labels.update(labels.numpy())

        # Convert label codes back to original labels if needed
        # This step is optional and depends on whether you need to display original label names
        # You would need a mapping from codes back to labels if labels were encoded as category codes

        print(f"Technology: {tech}")
        print(f"Total number of cells: {total_cells}")
        print(f"Unique labels: {unique_labels}\n")


def main():
    full, train_datasets, test, n_markers_dict = load_and_split_data(TECHS, OUTDIR, LABEL, LABEL_2)
    #full_dataset = load_full_data(TECHS, OUTDIR, LABEL, LABEL_2)
    full_dataset, label_mappings = load_full_data_and_label_mapping(TECHS, OUTDIR, LABEL, LABEL_2)
    trainer = initialize_models_and_training(full, n_markers_dict, first_source, TECHS, LABEL)
    trainer.restore(ckpt_dir)
    
    print_dataset_info(full_dataset)
    
    
    accumulated_data = []
    source_feature_cols = []
    target_feature_cols = []

    for (data, dlabel, _) in full_dataset[source]:
        source_mu, source_logvar = trainer.vae_lut[source].encode(data)
        source_codes = trainer.vae_lut[source].reparam(source_mu, source_logvar)

        source_translated_features = trainer.vae_lut[target].decode(source_codes)

        if not source_feature_cols:
            source_feature_cols = [f'{source}_feature_{i+1}' for i in range(data.shape[1])]
        if not target_feature_cols:
            target_feature_cols = [f'{target}_feature_{i+1}' for i in range(source_translated_features.shape[1])]

        data_np = data.numpy()
        source_translated_features_np = source_translated_features.numpy()
        dlabel_np = dlabel.numpy()
        dlabel_mapped = np.vectorize(label_mappings[source].get)(dlabel.numpy())

        for i in range(data_np.shape[0]):
            row = np.concatenate([[source], data_np[i], source_translated_features_np[i], [dlabel_np[i]], [dlabel_mapped[i]]])
            accumulated_data.append(row)

    for (data, dlabel, _) in full_dataset[target]:
        target_mu, target_logvar = trainer.vae_lut[target].encode(data)
        target_codes = trainer.vae_lut[target].reparam(target_mu, target_logvar)

        target_translated_features = trainer.vae_lut[source].decode(target_codes)

        data_np = data.numpy()
        target_translated_features_np = target_translated_features.numpy()
        dlabel_np = dlabel.numpy()
        dlabel_mapped = np.vectorize(label_mappings[source].get)(dlabel.numpy())

        for i in range(data_np.shape[0]):
            row = np.concatenate([[target], target_translated_features_np[i], data_np[i], [dlabel_np[i]], [dlabel_mapped[i]]])
            accumulated_data.append(row)

    columns = ['tech'] + source_feature_cols + target_feature_cols + ['label'] + ['original_label'] 

    df = pd.DataFrame(accumulated_data, columns=columns)
    # Save the DataFrame to a CSV file
    output_path = OUTDIR / 'integrated_features_and_labels.csv'
    df.to_csv(output_path, index=False)

    



if __name__ == "__main__":
    main()
  
    

    
  
