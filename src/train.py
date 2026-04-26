"""Training loop utilities for AE and VAE models.

Wraps Keras ``Model.fit`` with early stopping, learning-rate scheduling,
and model checkpointing.  Models are saved with a version tag so multiple
experimental runs can coexist in ``models/``.

Typical usage
-------------
>>> from src.train import set_seeds, train_ae, train_vae
>>> set_seeds(42)
>>> ae,  ae_hist  = train_ae('AbdomenCT',  train_ds, val_ds)
>>> vae, vae_hist = train_vae('AbdomenCT', train_ds, val_ds)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import random
from typing import Dict, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ── Local ─────────────────────────────────────────────────────────────────────
from src.model import Autoencoder, VAE


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Covers Python's ``random``, NumPy, and TensorFlow.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_ae(
    region: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    image_size: int = 64,
    latent_dim: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    models_dir: str = "models",
    seed: int = 42,
    version: str = "v1",
) -> Tuple[Autoencoder, Dict]:
    """Train a deterministic Autoencoder for a given anatomical region.

    Training callbacks applied:
    - ``ModelCheckpoint`` — saves the best ``val_loss`` weights.
    - ``EarlyStopping`` — stops after 5 epochs without improvement.
    - ``ReduceLROnPlateau`` — halves the LR after 3 stagnant epochs.

    Args:
        region: Region name used for checkpoint file naming.
        train_ds: Batched training dataset.
        val_ds: Batched validation dataset.
        image_size: Spatial resolution expected by the model.
        latent_dim: Size of the bottleneck latent vector.
        epochs: Maximum number of training epochs.
        lr: Initial Adam learning rate.
        models_dir: Directory where checkpoints are written.
        seed: Random seed for reproducibility.
        version: Version tag appended to the checkpoint filename
            (e.g. ``'v1'`` → ``ae_AbdomenCT_best_v1.keras``).

    Returns:
        A tuple ``(trained_model, history_dict)`` where *history_dict*
        contains per-epoch ``'loss'`` and ``'val_loss'`` lists.
    """
    set_seeds(seed)
    ae = Autoencoder(image_size=image_size, latent_dim=latent_dim)
    ae.compile(optimizer=keras.optimizers.Adam(lr))
    for batch in train_ds.take(1):
        ae(batch)  # build weights before checkpoint callback registers them

    ckpt_path = os.path.join(models_dir, f"ae_{region}_best_{version}.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]
    history = ae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return ae, history.history


def train_vae(
    region: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    image_size: int = 64,
    latent_dim: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    beta: float = 1.0,
    models_dir: str = "models",
    seed: int = 42,
    version: str = "v1",
) -> Tuple[VAE, Dict]:
    """Train a Variational Autoencoder for a given anatomical region.

    Monitors ``val_total_loss`` (reconstruction + β·KL) for all callbacks.

    Training callbacks applied:
    - ``ModelCheckpoint`` — saves the best ``val_total_loss`` weights.
    - ``EarlyStopping`` — stops after 5 epochs without improvement.
    - ``ReduceLROnPlateau`` — halves the LR after 3 stagnant epochs.

    Args:
        region: Region name used for checkpoint file naming.
        train_ds: Batched training dataset.
        val_ds: Batched validation dataset.
        image_size: Spatial resolution expected by the model.
        latent_dim: Dimensionality of the latent space.
        epochs: Maximum number of training epochs.
        lr: Initial Adam learning rate.
        beta: KL divergence weight (β-VAE coefficient).
        models_dir: Directory where checkpoints are written.
        seed: Random seed for reproducibility.
        version: Version tag appended to the checkpoint filename
            (e.g. ``'v1'`` → ``vae_AbdomenCT_best_v1.keras``).

    Returns:
        A tuple ``(trained_model, history_dict)`` where *history_dict*
        contains per-epoch ``'total_loss'``, ``'recon_loss'``,
        ``'kl_loss'``, and their ``val_*`` counterparts.
    """
    set_seeds(seed)
    vae = VAE(image_size=image_size, latent_dim=latent_dim, beta=beta)
    vae.compile(optimizer=keras.optimizers.Adam(lr))
    for batch in train_ds.take(1):
        vae(batch)  # build weights before checkpoint callback registers them

    ckpt_path = os.path.join(models_dir, f"vae_{region}_best_{version}.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_total_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_total_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_total_loss",
            mode="min",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]
    history = vae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return vae, history.history
