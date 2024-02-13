# Script used to integrate scRNA-seq and CyTOF data from the SCIM paper.

import tensorflow as tf
import numpy as np
import pandas as pd
import anndata
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
                            plot_training_curve, integrate_target_to_source, initialize_latent_space)

# Enable eager execution for TensorFlow
tf.compat.v1.enable_eager_execution()

# Define constants and paths
OUTDIR = Path('/cluster/scratch/hugifl/scRNA_self_integration')
first_source = 'scrna_2'
TECHS = ['scrna_2', 'scrna']
LABEL = 'cell_type'
iterations = 1
initialization_KL_beta = 0.001

def main():
    full, train_datasets, test, n_markers_dict = load_and_split_data(TECHS, OUTDIR, LABEL)
    perform_pca_and_plot(full, TECHS, OUTDIR, LABEL)

    # Initialize models and get the trainer instance
    trainer = initialize_models_and_training(full, n_markers_dict, first_source, TECHS, LABEL)

    # Initialize latent space using source technology
    ckpt_dir_initialization = OUTDIR / '1_init' / 'model'
    source = first_source
    target = TECHS[1] if source == TECHS[0] else TECHS[0]
    SEED = 42  # Define your seed

    try:
        trainer.restore(ckpt_dir_initialization)
    except AssertionError:
        initialize_latent_space(SEED, source, target, trainer, train_datasets, ckpt_dir_initialization, initialization_KL_beta, OUTDIR)
    else:
        print('Loaded checkpoint')
    perform_pca_and_plot_reconstructions(trainer, full[source], source, OUTDIR, LABEL)
    visualize_latent_space(trainer, full, source, OUTDIR, LABEL)
    visualize_discriminator_performance(trainer, test, OUTDIR, LABEL)
    error = 0/0

    target_technology = first_source
    source_technology = TECHS[1] if target_technology == TECHS[0] else TECHS[0]
    for i in range(iterations):
        iteration = i + 1
        iteration = integrate_target_to_source(trainer, full, train_datasets, test, target_technology, source_technology, iteration, OUTDIR, LABEL)
        #cache = OUTDIR.joinpath(f'2_integrate_{iteration}', 'evals.h5ad') 
        plot_integrated_latent_space(trainer, test, iteration, target_technology, TECHS, LABEL, OUTDIR)
        #target_technology = source_technology
        #source_technology = TECHS[1] if target_technology == TECHS[0] else TECHS[0]
    
if __name__ == "__main__":
    main()