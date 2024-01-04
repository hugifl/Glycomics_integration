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

# Enable eager execution for TensorFlow
tf.compat.v1.enable_eager_execution()

# Define constants and paths
OUTDIR = Path('/cluster/scratch/hugifl/TitrationII_only_high_var_genes')
TECHS = ['AB', 'lectin']
LABEL = 'celltype_final'

def get_number_of_genes(full):
    return {tech: full[tech].n_vars for tech in TECHS}

def load_and_split_data_old():
    full, train_datasets, test = {}, {}, {}
    for tech in TECHS:
        path = OUTDIR / f'{tech}.h5ad'
        data = anndata.read(path)

        # Split data
        train_idx = np.random.rand(len(data)) < 0.85
        train_data = data[train_idx]
        test_data = data[~train_idx]

        # Create TensorFlow datasets for training data with indices
        train_datasets[tech] = tf.data.Dataset.from_tensor_slices(
            (train_data.X, train_data.obs[LABEL].cat.codes, tf.range(len(train_data))))
        train_datasets[tech] = train_datasets[tech].batch(256, drop_remainder=True)

        # Keep test data in its original format
        test[tech] = test_data

        full[tech] = data

    return full, train_datasets, test

def load_and_split_data():
    full, train_datasets, test = {}, {}, {}
    for tech in TECHS:
        path = OUTDIR / f'{tech}.h5ad'
        data = anndata.read(path)

        # Ensure LABEL column is of categorical type
        if not pd.api.types.is_categorical_dtype(data.obs[LABEL]):
            data.obs[LABEL] = pd.Categorical(data.obs[LABEL])

        # Split data
        train_idx = np.random.rand(len(data)) < 0.85
        train_data = data[train_idx]
        test_data = data[~train_idx]

        # Create TensorFlow datasets for training data with indices
        train_datasets[tech] = tf.data.Dataset.from_tensor_slices(
            (train_data.X, train_data.obs[LABEL].cat.codes, tf.range(len(train_data))))
        train_datasets[tech] = train_datasets[tech].batch(256, drop_remainder=True)

        # Keep test data in its original format
        test[tech] = test_data

        full[tech] = data

    return full, train_datasets, test

def perform_pca_and_plot(data):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 2 columns
    for idx, key in enumerate(TECHS):
        # Perform PCA on the full dataset
        sc.tl.pca(data[key], svd_solver='arpack')

        # Plotting
        for jdx, color in enumerate(['celltype_final', 'conc']):
            ax = axes[idx, jdx]
            sc.pl.pca(data[key], color=color, ax=ax, show=False)
            ax.set_title(f'{key} {color}')

    plt.savefig(OUTDIR / 'PCA_plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def visualize_latent_space(trainer, full):
    source_tech = 'AB'  # Replace 'AB' with the technology you want to use as source

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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, key in enumerate(['celltype_final', 'conc']):
        sc.pl.pca(codes, color=key, ax=axes[0, idx], show=False)
        sc.pl.tsne(codes, color=key, ax=axes[1, idx], show=False)
    plt.savefig(OUTDIR / 'Initiated_latent_space_plot.png', dpi=300)
    plt.close(fig)

def visualize_discriminator_performance(trainer, test):
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



def get_label_categories(data):
    # Assuming all datasets have the same label categories
    first_tech = list(data.keys())[0]
    return data[first_tech].obs[LABEL].cat.categories

def setup_models(label_categories, ngenes_dict):
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

    # Setting up VAEs for each technology
    vae_lut = {}
    for tech in TECHS:
        ngenes = ngenes_dict[tech]
        encoder = make_network(
            doutput=2 * latent_dim,
            units=[32] * 2,
            dinput=ngenes,
            batch_norm=True, dropout=0.2,
            name=f'{tech}-encoder')

        decoder = make_network(
            doutput=ngenes,
            units=[32] * 2,
            dinput=latent_dim,
            batch_norm=True, dropout=0.2,
            name=f'{tech}-decoder')

        vae_lut[tech] = VAE(
            encoder_net=encoder,
            decoder_net=decoder,
            name=f'{tech}-vae')

    return vae_lut, discriminator

# Initialize models and training
def initialize_models_and_training(full, train, test):
    # Assuming LABEL is a global constant defined earlier
    label_categories = get_label_categories(full)

    # Get the number of genes
    ngenes = get_number_of_genes(full)

    # Setup models
    vae_lut, discriminator = setup_models(label_categories, ngenes)

    # Training logic
    genopts = {key: tf.keras.optimizers.Adam() for key in TECHS}
    disopt = tf.keras.optimizers.Adam()
    
    source_key = 'AB'
    trainer = Trainer(
        vae_lut=vae_lut,
        discriminator=discriminator,
        source_key=source_key,
        disopt=disopt,
        genopt_lut=genopts,
        beta=0.5)

    return trainer

def plot_training_curve(history):
    # Plot training curve logic
    fig, ax = plt.subplots()
    for key, pairs in history.items():
        step, vals = zip(*pairs)
        ax.plot(step, vals, label=key)
    ax.legend()
    #ax.set_ylim(0, 2)
    plt.savefig(OUTDIR / f'training_curve_initializing.png', dpi=300)
    plt.close(fig)


def integrate_target_to_source(trainer, full, train_datasets, test, integrated_technology='lectin'):
    ckpt_dir = OUTDIR / f'2_integrate' / 'model'
    source = 'AB'
    target = integrated_technology
    SEED = 42

    try:
        trainer.restore(ckpt_dir)
        print('Loaded checkpoint')
        # Check if history is available and plot the training curve
    except AssertionError:
        print('Training integration')
        np.random.seed(SEED)
        tf.compat.v1.random.set_random_seed(SEED)

        gs = 0
        for epoch in range(25):
            print(f'Epoch {epoch}')
            for (data, dlabel, didx), (prior, plabel, pidx) in zip(train_datasets[source], cycle(train_datasets[target])): 
                # Train the discriminator
                for _ in range(trainer.niters_discriminator):
                    disc_loss, _ = trainer.discriminator_step(source, target, data, prior, dlabel, plabel) 

                # Adversarial training
                loss, (nll, adv), (codes, recon) = trainer.adversarial_step(source, data, dlabel)

                if gs % 5 == 0:
                    # Evaluate training batch
                    batch = full[source][didx.numpy()]
                    trainer.forward(source, batch, LABEL)

                    pbatch = full[target][pidx.numpy()]
                    trainer.forward(target, pbatch, LABEL)

                    lut = {
                        f'mse-{source}': batch.obs['loss-mse'].mean(),
                        f'probs-{source}': batch.obs['probs-discriminator'].mean(),
                        f'probs-{target}': pbatch.obs['probs-discriminator'].mean(),
                        f'discriminator-{source}': batch.obs['loss-discriminator'].mean(),
                        f'discriminator-{target}': pbatch.obs['loss-discriminator'].mean()
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
        plot_training(trainer, axes)

    else:
        print('Loaded checkpoint')

def perform_bipartite_matching(inter, cache):
    source_pd = adata_to_pd(inter[inter.obs.tech=='AB'], add_cell_code_name='AB')
    target_pd = adata_to_pd(inter[inter.obs.tech=='lectin'], add_cell_code_name='lectin')

    # Build an extended knn graph with k = 10% cells
    G = get_cost_knn_graph(source_pd, target_pd, knn_k=20, null_cost_percentile=95, capacity_method='uniform')

    # Run mcmf and extract matches
    row_ind, col_ind = mcmf(G)
    matches = extract_matched_labels(inter[inter.obs.tech=='AB'], inter[inter.obs.tech=='lectin'],
                                     row_ind, col_ind, keep_cols=['conc','celltype_final'])
    
    cache.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(cache, index=False)

    print(f'Caching to {cache}')
    return matches


def main():
    print("--------------------------------- loading data ---------------------------------")
    full, train_datasets, test = load_and_split_data()
    print("--------------------------------- perform PCA ---------------------------------")
    perform_pca_and_plot(full)
    print("--------------------------------- initialize models and training ---------------------------------")
    # Initialize models and get the trainer instance
    trainer = initialize_models_and_training(full, train_datasets, test)

    # Initialize latent space using source technology
    ckpt_dir = OUTDIR / '1_init' / 'model'
    source = 'lectin' 
    target = 'AB'
    SEED = 42  # Define your seed

    try:
        trainer.restore(ckpt_dir)
    except AssertionError:
        print('---------------------------------------------- Initializing latent space by training VAE ----------------------------------------------')
        np.random.seed(SEED)
        tf.compat.v1.random.set_random_seed(SEED)

        gs = 0
        for epoch in range(10):       # increased epochs from 10
            print(f'Epoch {epoch}')
            for (data, dlabel, _), (prior, plabel, _) in zip(train_datasets[source], cycle(train_datasets[target])):
                # Initializes latent space
                loss, (mse, kl), (codes, recon) = trainer.vae_step(source, data, beta=0.001) # 0.001

                # Initialize the discriminator
                disc_loss, _ = trainer.discriminator_step(source, target, data, prior, dlabel, plabel)

                # Record
                if gs % 10 == 0:
                    lut = {'loss': loss.numpy().mean(), 'mse': mse.numpy().mean(), 'kl': kl.numpy().mean()}
                    trainer.record('vae', lut, step=gs)

                gs += 1

        trainer.saver.save(str(ckpt_dir / 'ckpt'))
        plot_training_curve(trainer.history['vae'])
    else:
        print('Loaded checkpoint')
    print("--------------------------------- visualize latent space ---------------------------------")
    visualize_latent_space(trainer, full)
    print("--------------------------------- visualize discriminator performance ---------------------------------")
    visualize_discriminator_performance(trainer, test)
    print("--------------------------------- integrate target to source ---------------------------------")
    integrate_target_to_source(trainer, full, train_datasets, test)

    # Integrated latent space visualization
    cache = OUTDIR.joinpath('2_integrate', 'evals.h5ad') # change manually
    try:
        evald = anndata.read(cache)
        print(f'Read from {cache}')
    except OSError:
        print('Cache loading failed')
        evald = trainer.evaluate(test, LABEL)
        sc.tl.pca(evald)
        sc.tl.tsne(evald, n_jobs=5)
        evald.write(cache)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'wspace':0.5})
    for color, ax in zip(['tech', 'celltype_final', 'probs-discriminator', 'conc'], axes.ravel()):
        kwargs = {}
        if color == 'probs-discriminator':
            kwargs = {'color_map': 'PiYG', 'vmin':0, 'vmax':1}
        sc.pl.tsne(evald, color=color, ax=ax, show=False, **kwargs)

    plt.savefig(OUTDIR / 'tsne_integrated_latent_space.png', dpi=300)
    plt.close(fig)

    # Cache model outputs
    trainer.forward(source, full[source], LABEL)
    trainer.forward(target, full[target], LABEL)

    scode = switch_obsm(full[source], 'code')
    tcode = switch_obsm(full[target], 'code')
    inter = scode.concatenate(tcode, batch_categories=[source, target], batch_key='tech')

    # Save the processed data
    full[source].write(OUTDIR.joinpath('2_integrate', f'{source}.h5ad'))
    full[target].write(OUTDIR.joinpath('2_integrate', f'{target}.h5ad'))
    inter.write(OUTDIR.joinpath('2_integrate', 'codes.h5ad'))

    # Bipartite matching
    cache = OUTDIR.joinpath('3_integrate', 'matches.csv')

    try:
        matches = pd.read_csv(cache)
        print(f'Read from cache {cache}')

    except OSError:
        print(f'Cache loading bipartite matching failed')
        matches = perform_bipartite_matching(inter, cache)

    # Evaluate matches based on branch label
    accuracy = get_accuracy(matches, colname_compare='celltype_final')
    print('accuracy (celltype_final): ', accuracy)
    confusion_matrix = get_confusion_matrix(matches, colname_compare='celltype_final')
    output_file_path = OUTDIR / 'confusion_matrix.csv'  # Specify your file path and name
    confusion_matrix.to_csv(output_file_path)

if __name__ == "__main__":
    main()