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
                            match_cells, evaluate_latent_space, plot_integrated_latent_space_full_data, match_cells2,
                            integrate_ADTs_to_RNA, reinitialize_training_with_new_discriminator, train_new_discriminator)

# Enable eager execution for TensorFlow
tf.compat.v1.enable_eager_execution()

# Define constants and paths
OUTDIR = Path('/cluster/scratch/hugifl/9_glycomics_8_RNA_regularized_4_3')
TECHS = ['RNA', 'AB','lectin']
ADT_TECHS = ['AB','lectin']
pre_integration_source = 'RNA'
first_ADT_source = 'AB'
LABEL = 'celltype_major'
LABEL_2 = 'celltype_final'
iterations_pre_integration = 5
iterations = 6
initialization_KL_beta = 0.005 # set to 0.001 for original experiment
epochs_pre_integration = 500

def main():
    full, train_datasets, test, n_markers_dict = load_and_split_data(TECHS, OUTDIR, LABEL, LABEL_2)
    perform_pca_and_plot(full, TECHS, OUTDIR, LABEL, n_techs=3)

    # Initialize models and get the trainer instance
    trainer = initialize_models_and_training(full, n_markers_dict, pre_integration_source, TECHS, LABEL)

    # Initialize latent space using source technology
    ckpt_dir_initialization = OUTDIR / '1_init' / 'model'
    target = first_ADT_source
    SEED = 42  # Define your seed

    try:
        trainer.restore(ckpt_dir_initialization)
    except AssertionError:
        initialize_latent_space(SEED, pre_integration_source, target, trainer, train_datasets, ckpt_dir_initialization, initialization_KL_beta, OUTDIR)
    else:
        print('Loaded checkpoint')
    perform_pca_and_plot_reconstructions(trainer, full[pre_integration_source], pre_integration_source, OUTDIR, LABEL)
    perform_pca_and_plot_reconstructions(trainer, full[pre_integration_source], pre_integration_source, OUTDIR, LABEL_2)
    visualize_latent_space(trainer, full, pre_integration_source, OUTDIR, LABEL)
    visualize_discriminator_performance(trainer, test, OUTDIR, LABEL)

    
    for i in range(iterations_pre_integration):
        print("pre integration iteration: ", i + 1)
        iteration = i + 1
        target_technology = pre_integration_source
        source_technology = first_ADT_source
        print("integrating ", source_technology, " to ", target_technology)
        iteration, cachedir = integrate_ADTs_to_RNA(epochs_pre_integration, trainer, full, train_datasets, test, target_technology, source_technology, iteration, OUTDIR, LABEL)
        #evaluate latent space
        div_score = evaluate_latent_space(test, trainer, LABEL)
        print("divergence score: ", div_score)
        out = OUTDIR.joinpath(f'divergence_score_pre_integration_{iteration}_{source_technology}_ADT_TECH.csv')
        with open(out, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([div_score])

        plot_integrated_latent_space(trainer, test, iteration, target_technology, source_technology, TECHS, LABEL, OUTDIR, LABEL2=LABEL_2, cachedir=cachedir, pre_training = True)
        plot_integrated_latent_space_full_data(trainer, full, iteration, target_technology, source_technology, TECHS, LABEL, OUTDIR, LABEL2=None, cachedir=cachedir, pre_training = True)
        inter = cache_model_outputs(trainer, full, target_technology, source_technology, OUTDIR, LABEL, iteration, target_technology, cachedir=cachedir)
        
        previous_source = source_technology
        source_technology = ADT_TECHS[1] if previous_source == ADT_TECHS[0] else ADT_TECHS[0]
        print("integrating ", source_technology, " to ", target_technology)
        iteration, cachedir = integrate_ADTs_to_RNA(epochs_pre_integration, trainer, full, train_datasets, test, target_technology, source_technology, iteration, OUTDIR, LABEL)
        #evaluate latent space
        div_score = evaluate_latent_space(test, trainer, LABEL)
        print("divergence score: ", div_score)
        out = OUTDIR.joinpath(f'divergence_score_pre_integration_{iteration}_{source_technology}_ADT_TECH.csv')
        with open(out, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([div_score])

        plot_integrated_latent_space(trainer, test, iteration, target_technology, source_technology, TECHS, LABEL, OUTDIR, LABEL2=LABEL_2, cachedir=cachedir, pre_training = True)
        plot_integrated_latent_space_full_data(trainer, full, iteration, target_technology, source_technology, TECHS, LABEL, OUTDIR, LABEL2=None, cachedir=cachedir, pre_training = True)
        inter = cache_model_outputs(trainer, full, target_technology, source_technology, OUTDIR, LABEL, iteration, target_technology, cachedir=cachedir)

    trained_vae_lut = trainer.get_trained_vae_lut()
    new_source_key = 'AB'
    new_trainer = reinitialize_training_with_new_discriminator(full, trained_vae_lut, TECHS, LABEL, new_source_key)

    source_technology = new_source_key
    target_technology = ADT_TECHS[1] if source_technology == ADT_TECHS[0] else ADT_TECHS[0]
    train_new_discriminator(new_trainer, source_technology, target_technology, train_datasets, OUTDIR, num_epochs=500) 
    
    for i in range(iterations):
        print("integrating ADT data")
        iteration = i + 1
        print("iteration: ", iteration, " out of ", iterations)
        print("integrating ", source_technology, " to ", target_technology)
        iteration = integrate_target_to_source(new_trainer, full, train_datasets, test, target_technology, source_technology, iteration, OUTDIR, LABEL)
        #evaluate latent space
        div_score = evaluate_latent_space(test, new_trainer, LABEL)
        print("divergence score: ", div_score)
        out = OUTDIR.joinpath(f'divergence_score_{iteration}.csv')
        with open(out, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([div_score])

        #visualize_latent_space(new_trainer, full, target_technology, OUTDIR, LABEL, iteration=iteration)
        #visualize_latent_space(new_trainer, full, source_technology, OUTDIR, LABEL, iteration=iteration)
        #cache = OUTDIR.joinpath(f'2_integrate_{iteration}', 'evals.h5ad') 
        plot_integrated_latent_space(new_trainer, test, iteration, target_technology, source_technology, TECHS, LABEL, OUTDIR, LABEL2=LABEL_2)
        plot_integrated_latent_space_full_data(new_trainer, full, iteration, target_technology, source_technology, TECHS, LABEL, OUTDIR, LABEL2=None)
        inter = cache_model_outputs(new_trainer, full, target_technology, source_technology, OUTDIR, LABEL, iteration, target_technology)
        target_technology = source_technology
        source_technology = ADT_TECHS[1] if target_technology == ADT_TECHS[0] else ADT_TECHS[0]
        match_cells(OUTDIR, source_technology, target_technology, inter, iteration = i)
    
    
if __name__ == "__main__":
    main()