import tensorflow as tf
import os
import numpy as np
import pandas as pd
import anndata
from itertools import cycle
import scanpy as sc
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from scim.utils import make_network
from scim.model import VAE, Integrator
from scim.discriminator import SpectralNormCritic
from scim.trainer import Trainer
from scim.utils import switch_obsm, make_network, plot_training, adata_to_pd
from scim.simulate import simulate
from scim.evaluate import score_divergence, extract_matched_labels, get_accuracy, get_confusion_matrix, get_correlation
from scim.matching import get_cost_knn_graph, mcmf


def get_number_of_genes(full, tech_no=0):
    # Assuming all datasets have the same number of genes
    first_tech = list(full.keys())[tech_no]
    print("number of genes: ", full[first_tech].n_vars)
    return full[first_tech].n_vars

def load_and_split_data(TECHS, OUTDIR, LABEL, LABEL_2=None):
    full, train_datasets, test = {}, {}, {}
    n_markers_dict = {}

    for tech in TECHS:
        path = OUTDIR / f'{tech}.h5ad'
        data = anndata.read(path)
        data.obs[LABEL] = data.obs[LABEL].astype('category')
        if LABEL_2 is not None:
            data.obs[LABEL_2] = data.obs[LABEL_2].astype('category')
        n_markers = data.X.shape[1]
        n_markers_dict[tech] = n_markers

        # Initialize masks
        train_idx = np.zeros(len(data), dtype=bool)
        test_idx = np.zeros(len(data), dtype=bool)

        # Split data for each cell type
        for cell_type in data.obs[LABEL].cat.categories:
            mask = data.obs[LABEL] == cell_type
            indices = data[mask].obs_names.tolist()  # Convert to a list for shuffling
            np.random.shuffle(indices)
            split_index = int(0.8 * len(indices))
            train_indices = indices[:split_index]
            test_indices = indices[split_index:]

            train_idx[data.obs_names.isin(train_indices)] = True
            test_idx[data.obs_names.isin(test_indices)] = True

        # Create training and test data
        train_data = data[train_idx]
        test_data = data[test_idx]

        # Create TensorFlow datasets for training data with indices
        train_datasets[tech] = tf.data.Dataset.from_tensor_slices(
            (train_data.X, train_data.obs[LABEL].cat.codes, tf.range(len(train_data))))
        train_datasets[tech] = train_datasets[tech].batch(256, drop_remainder=True)

        # Keep test data in its original format
        test[tech] = test_data

        full[tech] = data

    return full, train_datasets, test, n_markers_dict

def load_and_split_data_old(TECHS, OUTDIR, LABEL, LABEL_2=None):
    full, train_datasets, test = {}, {}, {}
    n_markers_dict = {}
    for tech in TECHS:
        path = OUTDIR / f'{tech}.h5ad'
        data = anndata.read(path)
        data.obs[LABEL] = data.obs[LABEL].astype('category')
        if not LABEL_2 is None:
            data.obs[LABEL_2] = data.obs[LABEL_2].astype('category')
        n_markers = data.X.shape[1]
        n_markers_dict[tech] = n_markers
        # Split data
        train_idx = np.random.rand(len(data)) < 0.6 # CHANGE BACK TO 0.6 if you want to use the original code
        train_data = data[train_idx]
        test_data = data[~train_idx]

        # Create TensorFlow datasets for training data with indices
        train_datasets[tech] = tf.data.Dataset.from_tensor_slices(
            (train_data.X, train_data.obs[LABEL].cat.codes, tf.range(len(train_data))))
        train_datasets[tech] = train_datasets[tech].batch(256, drop_remainder=True)

        # Keep test data in its original format
        test[tech] = test_data

        full[tech] = data

    return full, train_datasets, test, n_markers_dict

def perform_pca_and_plot(data, TECHS, OUTDIR, LABEL):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    for idx, key in enumerate(TECHS):
        # Perform PCA on the full dataset
        sc.tl.pca(data[key], svd_solver='arpack')

        # Convert to categorical type for visualization
        data[key].obs[LABEL] = data[key].obs[LABEL].astype('category')

        # Plotting PCA for each technology
        ax = axes[idx]  # Use index to select the subplot
        sc.pl.pca(data[key], color=LABEL, ax=ax, show=False)
        ax.set_title(f'{key} PCA')

    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig(OUTDIR / 'PCA_plot.png', dpi=300)
    plt.close(fig)

def perform_pca_and_plot_reconstructions(trainer, original_data, source_tech, outdir, LABEL):
    save_path = outdir / f'PCA_{source_tech}_reconstructions_{LABEL}.png'
    # Reconstruct the data
    trainer.forward(source_tech, original_data, LABEL)
    reconstructed_data = original_data.obsm['recon']

    reconstructed_adata = anndata.AnnData(X=reconstructed_data, obs=original_data.obs[[LABEL]])

    # Perform PCA on both original and reconstructed data
    sc.tl.pca(original_data, svd_solver='arpack')
    sc.tl.pca(reconstructed_adata, svd_solver='arpack')

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot PCA of original data
    sc.pl.pca(original_data, color=LABEL, ax=axes[0], title='Original Data PCA')

    # Plot PCA of reconstructed data
    sc.pl.pca(reconstructed_adata, color=LABEL, ax=axes[1], title='Reconstructed Data PCA')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def visualize_latent_space(trainer, full, source_tech, OUTDIR, LABEL, iteration=None):
    full_latent_space_dir = os.path.join(OUTDIR, 'full_latent_space')
    if not os.path.exists(full_latent_space_dir):
        os.makedirs(full_latent_space_dir)

    if iteration is None:
        path = OUTDIR / f'1_init_{source_tech}_codes.h5ad'
        path.parent.mkdir(exist_ok=True, parents=True)

        try:
            codes = anndata.read(path)
            print(f'Loaded from cache {path}')
        except OSError:
            print('Cache loading failed')
            trainer.forward(source_tech, full[source_tech], LABEL)
            codes = switch_obsm(full[source_tech], 'code')
            sc.tl.pca(codes)
            sc.tl.tsne(codes)
            codes.write(path)
            print(f'Caching to {path}')

        # Adjust subplot layout to 2 rows, 1 column
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Convert to categorical type for visualization
        codes.obs[LABEL] = codes.obs[LABEL].astype('category')

        # Plot PCA
        sc.pl.pca(codes, color=LABEL, ax=axes[0], show=False)
        axes[0].set_title(f'{source_tech} PCA')

        # Plot t-SNE
        sc.pl.tsne(codes, color=LABEL, ax=axes[1], show=False)
        axes[1].set_title(f'{source_tech} t-SNE')

        plt.savefig(OUTDIR / f'Initiated_latent_space_plot_{source_tech}.png', dpi=300)
        plt.close(fig)
    
    else:
        cache = OUTDIR / f'2_integration_iteration_{iteration}_{source_tech}' / 'evals.h5ad' 
        try:
            evald = anndata.read(cache)
            print(f'Read from {cache}')
        except OSError:
            print('Cache loading failed')
            trainer.forward(source_tech, full[source_tech], LABEL)
            codes = switch_obsm(full[source_tech], 'code')
            sc.tl.pca(codes)
            sc.tl.tsne(codes)
            codes.write(cache)
            print(f'Caching to {cache}')

        # Adjust subplot layout to 2 rows, 1 column
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Convert to categorical type for visualization
        codes.obs[LABEL] = codes.obs[LABEL].astype('category')

        # Plot PCA
        sc.pl.pca(codes, color=LABEL, ax=axes[0], show=False)
        axes[0].set_title(f'{source_tech} PCA')

        # Plot t-SNE
        sc.pl.tsne(codes, color=LABEL, ax=axes[1], show=False)
        axes[1].set_title(f'{source_tech} t-SNE')

        plt.savefig(OUTDIR / f'latent_space_plot_{source_tech}_{iteration}.png', dpi=300)
        plt.close(fig)

def visualize_discriminator_performance(trainer, test, OUTDIR, LABEL):
    # Path for caching the evaluation results
    path = OUTDIR / '1_init' / 'evals.h5ad'
    path.parent.mkdir(exist_ok=True, parents=True)

    try:
        # Try loading the evaluation data from cache
        evald = anndata.read(path)
        print(f'Loaded from cache {path}')
    except OSError:
        # Cache loading failed, perform evaluation
        print('Cache loading failed')
        evald = trainer.evaluate(test, LABEL)
        sc.tl.tsne(evald)
        evald.write(path)
        print(f'Caching to {path}')

    # Plot t-SNE visualizations for discriminator probabilities
    ax = sc.pl.tsne(evald, color='probs-discriminator', color_map='PiYG', vmin=0, vmax=1, show=False)
    fig = ax.get_figure()
    fig.savefig(OUTDIR / 'tsne_discriminator_probs.png', dpi=300, bbox_inches='tight')

    # Plot t-SNE visualizations for technology
    ax = sc.pl.tsne(evald, color='tech', show=False)
    fig = ax.get_figure()
    fig.savefig(OUTDIR / 'tsne_tech.png', dpi=300, bbox_inches='tight')

def get_label_categories(data, LABEL):
    # Assuming both datasets have the same categories
    first_tech = list(data.keys())[0]
    categories_first_tech = pd.Categorical(data[first_tech].obs[LABEL]).categories
    return categories_first_tech

def setup_models(label_categories, n_markers_dict, TECHS):
    
    latent_dim = 8

    # Setting up the discriminator
    discriminator_net = make_network(
        doutput=1,
        units=[8] * 2,
        dinput=latent_dim + len(label_categories),
        batch_norm=False,
        name='discriminator')
    
    discriminator = SpectralNormCritic(
        discriminator_net,
        input_cats=label_categories)

    vae_lut = {}
    for tech in TECHS:
        nunits = 64 # 8 if tech == 'cytof' else 128          ###### CHANGE BACK TO 64 if you want to use the original code
        encoder = make_network(
            doutput=2 * latent_dim,
            units=[nunits] * 2,                         ###### CHANGE BACK TO 2 if you want to use the original code
            dinput=n_markers_dict[tech],
            batch_norm=True, dropout=0.2,
            name=f'{tech}-encoder')

        decoder = make_network(
            doutput=n_markers_dict[tech],
            units=[nunits] * 2,
            dinput=latent_dim,
            batch_norm=True, dropout=0.2,
            name=f'{tech}-decoder')

        vae_lut[tech] = VAE(
            encoder_net=encoder,
            decoder_net=decoder,
            name=f'{tech}-vae')

    return vae_lut, discriminator

def plot_integrated_latent_space_old(trainer, test, iteration, target_technology, TECHS, LABEL, OUTDIR, LABEL2=None):
    print(f"Plotting latent space after {iteration} iterations of integration")
    cache = OUTDIR / f'2_integration_iteration_{iteration}_{target_technology}' / 'evals.h5ad' 
    try:
        evald = anndata.read(cache)
        print(f'Read from {cache}')
    except OSError:
        print('Cache loading failed')
        evald = trainer.evaluate(test, LABEL)
        evald.obs[LABEL] = evald.obs[LABEL].astype('category')
        sc.tl.pca(evald)
        sc.tl.tsne(evald, n_jobs=5)
        evald.write(cache)

    kwargs = {}
    fig, axes = plt.subplots(1, 3, figsize=(36, 8))  # Create a figure with 1 row and 3 columns of subplots

    sc.pl.tsne(evald, color=LABEL, ax=axes[0], show=False)
    axes[0].set_title('t-SNE colored by Cell Type')
    sc.pl.tsne(evald, color='tech', ax=axes[1], show=False)
    axes[1].set_title('t-SNE colored by Tech')
    sc.pl.tsne(evald, color='probs-discriminator', ax=axes[2], show=False, color_map='PiYG', vmin=0, vmax=1)
    axes[2].set_title('t-SNE colored by Probs-Discriminator')

    plt.savefig(OUTDIR / f'tsne_integrated_latent_space_full_iter_{iteration}.png', dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 16))  
    for key, ax in zip(TECHS, axes):  
        mask = evald.obs['tech'] == key
        sc.pl.tsne(evald[mask], color=LABEL, ax=ax, show=False, **kwargs)
        ax.set_title(f't-SNE for {key}')
    plt.savefig(OUTDIR / f'tsne_integrated_latent_space_separate_iter_{iteration}.png', dpi=300)
    plt.close(fig)

    if not LABEL2 is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 16))  
        for key, ax in zip(TECHS, axes):  
            mask = evald.obs['tech'] == key
            sc.pl.tsne(evald[mask], color=LABEL2, ax=ax, show=False, **kwargs)
            ax.set_title(f't-SNE for {key}')
        plt.savefig(OUTDIR / f'tsne_integrated_latent_space_separate_iter_{iteration}_{LABEL2}.png', dpi=300)
        plt.close(fig)

def plot_integrated_latent_space(trainer, test, iteration, target_technology, TECHS, LABEL, OUTDIR, LABEL2=None):
    print(f"Plotting latent space after {iteration} iterations of integration")
    cache = OUTDIR / f'2_integration_iteration_{iteration}_{target_technology}' / 'evals.h5ad' 
    try:
        evald = anndata.read(cache)
        print(f'Read from {cache}')
    except OSError:
        print('Cache loading failed')
        evald = trainer.evaluate(test, LABEL)
        evald.obs[LABEL] = evald.obs[LABEL].astype('category')
        sc.tl.pca(evald)
        sc.tl.tsne(evald, n_jobs=5)
        sc.pp.neighbors(evald, use_rep='X_pca')
        #sc.tl.umap(evald)
        evald.write(cache)


    kwargs = {}
    fig, axes = plt.subplots(1, 3, figsize=(36, 8))  # Create a figure with 1 row and 3 columns of subplots

    sc.pl.tsne(evald, color=LABEL, ax=axes[0], show=False)
    axes[0].set_title('t-SNE colored by Cell Type')
    sc.pl.tsne(evald, color='tech', ax=axes[1], show=False)
    axes[1].set_title('t-SNE colored by Tech')
    sc.pl.tsne(evald, color='probs-discriminator', ax=axes[2], show=False, color_map='PiYG', vmin=0, vmax=1)
    axes[2].set_title('t-SNE colored by Probs-Discriminator')

    plt.savefig(OUTDIR / f'tsne_integrated_latent_space_full_iter_{iteration}.png', dpi=300)
    plt.close(fig)

    # Plotting for each method: PCA, t-SNE, and UMAP
    for plot_method in ['pca', 'tsne']:
        fig, axes = plt.subplots(2, 1, figsize=(12, 16))
        for key, ax in zip(TECHS, axes):
            mask = evald.obs['tech'] == key
            if plot_method == 'pca':
                sc.pl.pca(evald[mask], color=LABEL, ax=ax, show=False)
            elif plot_method == 'tsne':
                sc.pl.tsne(evald[mask], color=LABEL, ax=ax, show=False)
            #else:  # UMAP
            #    sc.pl.umap(evald[mask], color=LABEL, ax=ax, show=False)
            ax.set_title(f'{plot_method.upper()} for {key}')
        plt.savefig(OUTDIR / f'{plot_method}_integrated_latent_space_separate_iter_{iteration}_{LABEL}.png', dpi=300)
        plt.close(fig)

        if LABEL2 is not None:
            fig, axes = plt.subplots(2, 1, figsize=(12, 16))
            for key, ax in zip(TECHS, axes):
                mask = evald.obs['tech'] == key
                if plot_method == 'pca':
                    sc.pl.pca(evald[mask], color=LABEL2, ax=ax, show=False)
                elif plot_method == 'tsne':
                    sc.pl.tsne(evald[mask], color=LABEL2, ax=ax, show=False)
                #else:  # UMAP
                #    sc.pl.umap(evald[mask], color=LABEL2, ax=ax, show=False)
                ax.set_title(f'{plot_method.upper()} for {key} ({LABEL2})')
            plt.savefig(OUTDIR / f'{plot_method}_integrated_latent_space_separate_iter_{iteration}_{LABEL2}.png', dpi=300)
            plt.close(fig)

def initialize_models_and_training(full, n_markers_dict, first_source, TECHS, LABEL):
    # Assuming LABEL is a global constant defined earlier
    label_categories = get_label_categories(full, LABEL)

    # Setup models
    vae_lut, discriminator = setup_models(label_categories, n_markers_dict, TECHS)

    # Training logic
    genopts = {key: tf.keras.optimizers.Adam(learning_rate=0.0005) for key in TECHS}
    disopt = tf.keras.optimizers.Adam(learning_rate=0.0005)

    trainer = Trainer(
        vae_lut=vae_lut,
        discriminator=discriminator,
        source_key=first_source,
        disopt=disopt,
        genopt_lut=genopts,
        beta=0.08) # set to 0.01 for original experiment

    return trainer

def plot_training_curve(history, OUTDIR):
    # Plot training curve logic
    fig, ax = plt.subplots()
    for key, pairs in history.items():
        step, vals = zip(*pairs)
        ax.plot(step, vals, label=key)
    ax.legend()
    plt.savefig(OUTDIR / f'training_curve_initializing.png', dpi=300)
    plt.close(fig)

def integrate_target_to_source(trainer, full, train_datasets, test, target_technology, source_technology, iteration, OUTDIR, LABEL):
    ckpt_dir = OUTDIR / f'2_integration_iteration_{iteration}_{target_technology}' / 'model'
    SEED = 42

    try:
        trainer.restore(ckpt_dir)
        print('Loaded checkpoint')
    except AssertionError:
        print('Training integration')
        np.random.seed(SEED)
        tf.compat.v1.random.set_random_seed(SEED)

        gs = 0
        for epoch in range(int(500)):
            print(f'Epoch {epoch}')
            for (data, dlabel, didx), (prior, plabel, pidx) in zip(train_datasets[source_technology], cycle(train_datasets[target_technology])): 
                # Train the discriminator
                for _ in range(trainer.niters_discriminator):
                    disc_loss, _ = trainer.discriminator_step(source_technology, target_technology, data, prior, dlabel, plabel) 

                # Adversarial training
                loss, (nll, adv), (codes, recon) = trainer.adversarial_step(source_technology, data, dlabel)

                if gs % 5 == 0:
                    # Evaluate training batch
                    batch = full[source_technology][didx.numpy()]
                    trainer.forward(source_technology, batch, LABEL)

                    pbatch = full[target_technology][pidx.numpy()]
                    trainer.forward(target_technology, pbatch, LABEL)

                    lut = {
                        f'mse-{source_technology}': batch.obs['loss-mse'].mean(),
                        f'probs-{source_technology}': batch.obs['probs-discriminator'].mean(),
                        f'probs-{target_technology}': pbatch.obs['probs-discriminator'].mean(),
                        f'discriminator-{source_technology}': batch.obs['loss-discriminator'].mean(),
                        f'discriminator-{target_technology}': pbatch.obs['loss-discriminator'].mean()
                    }
                    trainer.record('train', lut, gs)

                if gs % 50 == 0:
                    # Evaluate test set
                    evald = trainer.evaluate(test, LABEL)
                    probs = evald.obs.groupby('tech')['probs-discriminator'].mean()
                    dloss = evald.obs.groupby('tech')['loss-discriminator'].mean()
                    mse = evald.obs.groupby('tech')['loss-mse'].mean()

                    lut = {'divergence': evald.uns['divergence']}
                    lut.update({f'probs-{k}': v for k, v in probs.to_dict().items()})
                    lut.update({f'mse-{k}': v for k, v in mse.to_dict().items()})
                    lut.update({f'discriminator-{k}': v for k, v in dloss.to_dict().items()})

                    trainer.record('test', lut, gs)

                gs += 1

        trainer.saver.save(str(ckpt_dir / 'ckpt'))
        fig, axes = plt.subplots(3, 1, figsize=(5, 9))
        plot_training(trainer, axes, OUTDIR)
        return iteration

    else:
        print('Loaded checkpoint')
        return iteration

def initialize_latent_space(SEED, source, target, trainer, train_datasets, ckpt_dir, init_KL_beta, OUTDIR):
    print('Initializing latent space by training VAE')
    np.random.seed(SEED)
    tf.compat.v1.random.set_random_seed(SEED)
    gs = 0
    for epoch in range(500):
        print(f'Epoch {epoch}')
        for (data, dlabel, _), (prior, plabel, _) in zip(train_datasets[source], cycle(train_datasets[target])):
            # Initializes latent space
            loss, (mse, kl), (codes, recon) = trainer.vae_step(source, data, beta= init_KL_beta)
            # Initialize the discriminator
            disc_loss, _ = trainer.discriminator_step(source, target, data, prior, dlabel, plabel)
            # Record
            if gs % 10 == 0:
                lut = {'loss': loss.numpy().mean(), 'mse': mse.numpy().mean(), 'kl': kl.numpy().mean()}
                trainer.record('vae', lut, step=gs)
            gs += 1
    trainer.saver.save(str(ckpt_dir / 'ckpt'))
    plot_training_curve(trainer.history['vae'], OUTDIR)

def cache_model_outputs(trainer, full, source, target, OUTDIR, LABEL, iteration, target_technology):
    trainer.forward(source, full[source], LABEL)
    trainer.forward(target, full[target], LABEL)

    scode = switch_obsm(full[source], 'code')
    tcode = switch_obsm(full[target], 'code')
    inter = scode.concatenate(tcode,
                              batch_categories=[source, target],
                              batch_key='tech')


    full[source].write(OUTDIR.joinpath(f'2_integration_iteration_{iteration}_{target_technology}', f'{source}.h5ad'))
    full[target].write(OUTDIR.joinpath(f'2_integration_iteration_{iteration}_{target_technology}', f'{target}.h5ad'))
    inter.write(OUTDIR.joinpath(f'2_integration_iteration_{iteration}_{target_technology}', f'codes.h5ad'))
    return inter

def match_cells(OUTDIR, source, target, inter):
    cache = OUTDIR.joinpath('3_integrate', 'matches.csv')

    try:
        matches = pd.read_csv(cache)
        print(f'Read from cache {cache}')

    except OSError:
        print(f'Cache loading failed')
        # Perform bipartite matching on the latent embeddings
        # This might take 5-10 mins
        source_pd = adata_to_pd(inter[inter.obs.tech==source], add_cell_code_name=source)
        target_pd = adata_to_pd(inter[inter.obs.tech==target], add_cell_code_name=target)

        # Build an extended knn graph with k = 10% cells
        G = get_cost_knn_graph(source_pd, target_pd, knn_k=20, null_cost_percentile=95, capacity_method='uniform')

        # Run mcmf and extract matches
        row_ind, col_ind = mcmf(G)
        matches = extract_matched_labels(inter[inter.obs.tech==source], inter[inter.obs.tech==target],
                                         row_ind, col_ind, keep_cols=['celltype_major','celltype_final'])

        cache.parent.mkdir(parents=True, exist_ok=True)
        matches.to_csv(cache, index=False)

        print(f'Caching to {cache}')
    
    accuracy, n_tp, n_matches_min_n_tp = get_accuracy(matches, colname_compare='celltype_major')
    print('accuracy (celltype_major): ', accuracy)
    confusion_matrix = get_confusion_matrix(matches, colname_compare='celltype_major')
    print('confusion matrix (celltype_major): ', confusion_matrix)
    #save confusion matrix
    confusion_matrix.to_csv(OUTDIR / 'confusion_matrix.csv', index=False)

def evaluate_latent_space(test, trainer, LABEL):
    codes = []
    sources = []

    # Iterate through each technology and its corresponding dataset
    for tech, dataset in test.items():
        # Forward pass through the model to get latent representations
        trainer.forward(tech, dataset, LABEL)

        # Extract and store the codes
        codes.append(dataset.obsm['code'])  # Replace 'code' with the key used in your model

        # Create a source array indicating the technology for each cell
        source_array = np.full(dataset.n_obs, tech)
        sources.append(source_array)

    # Concatenate all codes and sources
    codes = np.concatenate(codes, axis=0)
    sources = np.concatenate(sources, axis=0)

    # Make sure sources are numerical for scoring
    unique_sources, sources = np.unique(sources, return_inverse=True)

    # Calculate the divergence score
    div_score = score_divergence(codes, sources, k=50)
    return div_score
