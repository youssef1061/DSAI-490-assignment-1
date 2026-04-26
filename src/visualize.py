"""Visualisation utilities for AE and VAE experiments.

Provides functions for:
- Loss curve plots (AE and VAE)
- Original vs reconstructed image grids
- Native 2-D latent space scatter (VAE with latent_dim=2)
- PCA 2-D / 3-D and t-SNE 2-D latent space projections
- VAE generative sample grids
- Denoising comparison grids (clean / noisy / denoised)

All figures are saved to ``<output_dir>/<region>/`` and displayed inline.

Typical usage
-------------
>>> from src.visualize import plot_ae_loss, plot_reconstructions
>>> plot_ae_loss(ae_hist, region='AbdomenCT', output_dir=OUTPUT_DIR)
>>> plot_reconstructions(ae, val_ds, model_name='AE', region='AbdomenCT',
...                      output_dir=OUTPUT_DIR)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
from typing import Dict

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Local ─────────────────────────────────────────────────────────────────────
from src.model import Autoencoder, VAE


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(
    fig: plt.Figure,
    fname: str,
    region: str,
    output_dir: str,
) -> None:
    """Save *fig* to a region-specific subdirectory and display inline.

    Args:
        fig: Matplotlib figure to save.
        fname: Output filename (e.g. ``'ae_loss_curves.png'``).
        region: Region name — determines the output subdirectory.
        output_dir: Root output directory.
    """
    rdir = os.path.join(output_dir, region)
    os.makedirs(rdir, exist_ok=True)
    fig.savefig(os.path.join(rdir, fname), bbox_inches="tight", dpi=130)
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Loss curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_ae_loss(ae_hist: Dict, region: str, output_dir: str) -> None:
    """Plot AE training and validation MSE loss curves.

    Args:
        ae_hist: History dictionary returned by ``keras.Model.fit``.
        region: Region label used in the plot title and saved filename.
        output_dir: Root directory for saved figures.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ae_hist["loss"],     label="train loss")
    ax.plot(ae_hist["val_loss"], label="val loss", linestyle="--")
    ax.set_title(f"AE Loss Curves — {region}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "ae_loss_curves.png", region, output_dir)


def plot_vae_loss(vae_hist: Dict, region: str, output_dir: str) -> None:
    """Plot VAE training curves for total, reconstruction, and KL losses.

    Args:
        vae_hist: History dictionary returned by ``keras.Model.fit``.
        region: Region label used in the plot title and saved filename.
        output_dir: Root directory for saved figures.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(vae_hist["total_loss"],     label="train total")
    ax.plot(vae_hist["val_total_loss"], label="val total",  linestyle="--")
    ax.plot(vae_hist["recon_loss"],     label="recon",      linestyle=":")
    ax.plot(vae_hist["kl_loss"],        label="KL",         linestyle="-.")
    ax.set_title(f"VAE Loss Curves — {region}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "vae_loss_curves.png", region, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Reconstructions
# ─────────────────────────────────────────────────────────────────────────────

def plot_reconstructions(
    model: tf.keras.Model,
    ds: tf.data.Dataset,
    model_name: str,
    region: str,
    output_dir: str,
    n: int = 8,
) -> None:
    """Display and save original vs reconstructed images side by side.

    Args:
        model: Trained AE or VAE model.
        ds: Dataset to sample images from (first batch used).
        model_name: ``'AE'`` or ``'VAE'`` — used in the title and filename.
        region: Region label used in the plot title and saved filename.
        output_dir: Root directory for saved figures.
        n: Number of image pairs to show.
    """
    for batch in ds.take(1):
        imgs = batch[:n]
    recons = model(imgs, training=False)
    if isinstance(model, VAE):
        recons = recons[0]

    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.2))
    for i in range(n):
        axes[0, i].imshow(imgs[i, ..., 0].numpy(),   cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i, ..., 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original",      fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    fig.suptitle(f"{model_name} Reconstructions — {region}", fontsize=11, y=1.01)
    plt.tight_layout()
    _save(fig, f"{model_name.lower()}_reconstructions.png", region, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Latent space — native 2-D (VAE with latent_dim=2 only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_latent_space(
    vae: VAE,
    ds: tf.data.Dataset,
    region: str,
    output_dir: str,
    n_batches: int = 15,
) -> None:
    """Scatter-plot the native 2-D VAE latent space (only when latent_dim=2).

    Silently returns if ``latent_dim != 2``.

    Args:
        vae: Trained VAE with ``latent_dim=2``.
        ds: Validation dataset.
        region: Region label used in the plot title and saved filename.
        output_dir: Root directory for saved figures.
        n_batches: Maximum number of batches to encode.
    """
    z_all = np.concatenate(
        [vae.encoder(b, training=False)[0].numpy()
         for i, b in enumerate(ds) if i < n_batches],
        axis=0,
    )
    if z_all.shape[1] != 2:
        print(f"latent_dim={z_all.shape[1]} — skipping native 2-D scatter")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(z_all[:, 0], z_all[:, 1], s=7, alpha=0.45, c="steelblue")
    ax.set_title(f"VAE Latent Space 2-D (native) — {region}")
    ax.set_xlabel("z₀")
    ax.set_ylabel("z₁")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "vae_latent_space_native2d.png", region, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Sample generation (VAE only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_generated(vae: VAE, region: str, output_dir: str, n: int = 16) -> None:
    """Sample *n* images from the VAE prior and display them in a grid.

    Args:
        vae: Trained VAE model.
        region: Region label used in the plot title and saved filename.
        output_dir: Root directory for saved figures.
        n: Number of images to generate.
    """
    imgs = vae.sample(n).numpy()
    cols, rows = 8, (n + 7) // 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6))
    for i, ax in enumerate(np.array(axes).reshape(-1)):
        if i < n:
            ax.imshow(imgs[i, ..., 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(f"VAE Generated Samples — {region}", fontsize=11)
    plt.tight_layout()
    _save(fig, "vae_generated.png", region, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Denoising
# ─────────────────────────────────────────────────────────────────────────────

def plot_denoising(
    model: tf.keras.Model,
    ds: tf.data.Dataset,
    model_name: str,
    region: str,
    output_dir: str,
    noise_std: float = 0.3,
    n: int = 8,
) -> None:
    """Show clean / noisy / denoised image rows and save the figure.

    Args:
        model: Trained AE or VAE model.
        ds: Dataset to sample clean images from.
        model_name: ``'AE'`` or ``'VAE'`` — used in the title and filename.
        region: Region label used in the plot title and saved filename.
        output_dir: Root directory for saved figures.
        noise_std: Standard deviation of additive Gaussian noise.
        n: Number of image triplets to display.
    """
    for batch in ds.take(1):
        clean = batch[:n]
    noisy    = tf.clip_by_value(clean + tf.random.normal(clean.shape, stddev=noise_std), 0, 1)
    denoised = model(noisy, training=False)
    if isinstance(model, VAE):
        denoised = denoised[0]

    fig, axes = plt.subplots(3, n, figsize=(n * 1.6, 4.8))
    for i in range(n):
        axes[0, i].imshow(clean[i, ..., 0].numpy(),    cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(noisy[i, ..., 0].numpy(),    cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        axes[2, i].imshow(denoised[i, ..., 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[2, i].axis("off")
    for row, label in enumerate(["Clean", "Noisy", "Denoised"]):
        axes[row, 0].set_ylabel(label, fontsize=9)
    fig.suptitle(f"{model_name} Denoising — {region}", fontsize=11, y=1.01)
    plt.tight_layout()
    _save(fig, f"{model_name.lower()}_denoising.png", region, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Latent extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_ae_latents(
    ae: Autoencoder,
    ds: tf.data.Dataset,
    n_batches: int = 15,
) -> np.ndarray:
    """Encode *n_batches* of images through the AE encoder.

    Args:
        ae: Trained Autoencoder.
        ds: Dataset to extract latent representations from.
        n_batches: Maximum number of batches to process.

    Returns:
        NumPy array of shape ``(N, latent_dim)``.
    """
    return np.concatenate(
        [ae.encoder(b, training=False).numpy()
         for i, b in enumerate(ds) if i < n_batches],
        axis=0,
    )


def get_vae_latents(
    vae: VAE,
    ds: tf.data.Dataset,
    n_batches: int = 15,
) -> np.ndarray:
    """Encode *n_batches* of images through the VAE encoder (z_mean only).

    Using the posterior mean (rather than a sample) gives a deterministic,
    lower-variance representation suitable for visualisation.

    Args:
        vae: Trained VAE.
        ds: Dataset to extract latent representations from.
        n_batches: Maximum number of batches to process.

    Returns:
        NumPy array of shape ``(N, latent_dim)`` — the posterior means.
    """
    return np.concatenate(
        [vae.encoder(b, training=False)[0].numpy()
         for i, b in enumerate(ds) if i < n_batches],
        axis=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2-D / 3-D latent space projections (PCA and t-SNE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_2d(
    z: np.ndarray,
    model_name: str,
    region: str,
    output_dir: str,
) -> None:
    """Reduce latent vectors to 2-D via PCA and plot as a scatter.

    Args:
        z: Latent array of shape ``(N, latent_dim)``.
        model_name: ``'AE'`` or ``'VAE'`` — used in the title and filename.
        region: Region label used in the plot title and saved filename.
        output_dir: Root directory for saved figures.
    """
    coords = PCA(n_components=2).fit_transform(z)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=7, alpha=0.45, c="steelblue")
    ax.set_title(f"{model_name} Latent Space — PCA 2D — {region}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{model_name.lower()}_pca2d.png", region, output_dir)


def plot_tsne_2d(
    z: np.ndarray,
    model_name: str,
    region: str,
    output_dir: str,
    seed: int = 42,
) -> None:
    """Reduce latent vectors to 2-D via t-SNE and plot as a scatter.

    Args:
        z: Latent array of shape ``(N, latent_dim)``.
        model_name: ``'AE'`` or ``'VAE'`` — used in the title and filename.
        region: Region label used in the plot title and saved filename.
        output_dir: Root directory for saved figures.
        seed: Random state passed to ``TSNE`` for reproducibility.
    """
    coords = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(z)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=7, alpha=0.45, c="tomato")
    ax.set_title(f"{model_name} Latent Space — t-SNE 2D — {region}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, f"{model_name.lower()}_tsne2d.png", region, output_dir)


def plot_pca_3d(
    z: np.ndarray,
    model_name: str,
    region: str,
    output_dir: str,
) -> None:
    """Reduce latent vectors to 3-D via PCA and plot as a 3-D scatter.

    Args:
        z: Latent array of shape ``(N, latent_dim)``.
        model_name: ``'AE'`` or ``'VAE'`` — used in the title and filename.
        region: Region label used in the plot title and saved filename.
        output_dir: Root directory for saved figures.
    """
    coords = PCA(n_components=3).fit_transform(z)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=5, alpha=0.4, c="mediumseagreen")
    ax.set_title(f"{model_name} Latent Space — PCA 3D — {region}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    _save(fig, f"{model_name.lower()}_pca3d.png", region, output_dir)
