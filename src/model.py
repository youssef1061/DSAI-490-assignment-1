"""Autoencoder (AE) and Variational Autoencoder (VAE) model definitions.

Both models use a symmetric convolutional encoder / decoder architecture.
The VAE extends the AE with a probabilistic latent space and a KL-regularised
ELBO objective (β-VAE formulation).

Typical usage
-------------
>>> from src.model import Autoencoder, VAE
>>> ae  = Autoencoder(image_size=64, latent_dim=64)
>>> vae = VAE(image_size=64, latent_dim=64, beta=1.0)
"""

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Dict, List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

def ae_encoder(image_size: int = 64, latent_dim: int = 64) -> keras.Model:
    """Build the AE encoder: image → latent vector.

    Architecture: three strided Conv2D blocks (32 → 64 → 128 filters),
    followed by Flatten and a Dense projection to *latent_dim*.

    Args:
        image_size: Spatial input resolution (square).
        latent_dim: Dimensionality of the bottleneck vector.

    Returns:
        A ``keras.Model`` with output shape ``(batch, latent_dim)``.
    """
    inp = keras.Input(shape=(image_size, image_size, 1))
    x = layers.Conv2D(32,  3, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    z = layers.Dense(latent_dim, name="latent")(x)
    return keras.Model(inp, z, name="ae_encoder")


def ae_decoder(image_size: int = 64, latent_dim: int = 64) -> keras.Model:
    """Build the AE decoder: latent vector → reconstructed image.

    Architecture: Dense projection + Reshape, followed by three strided
    Conv2DTranspose blocks (128 → 64 → 32 filters) and a sigmoid output Conv.

    Args:
        image_size: Spatial output resolution (square).
        latent_dim: Dimensionality of the input bottleneck vector.

    Returns:
        A ``keras.Model`` with output shape ``(batch, image_size, image_size, 1)``.
    """
    h = image_size // 8
    inp = keras.Input(shape=(latent_dim,))
    x = layers.Dense(h * h * 128, activation="relu")(inp)
    x = layers.Reshape((h, h, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32,  3, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv2D(1, 3, padding="same", activation="sigmoid", name="reconstruction")(x)
    return keras.Model(inp, out, name="ae_decoder")


class Autoencoder(keras.Model):
    """Deterministic convolutional Autoencoder trained with pixel-wise MSE loss.

    Attributes:
        encoder: Encoder sub-model — maps images to latent vectors.
        decoder: Decoder sub-model — maps latent vectors back to images.
        loss_tracker: Running mean of the per-step MSE reconstruction loss.
    """

    def __init__(
        self,
        image_size: int = 64,
        latent_dim: int = 64,
        **kwargs,
    ) -> None:
        """Initialise encoder, decoder, and loss tracker.

        Args:
            image_size: Spatial resolution of input images.
            latent_dim: Size of the bottleneck representation.
            **kwargs: Forwarded to ``keras.Model.__init__``.
        """
        super().__init__(**kwargs)
        self.encoder = ae_encoder(image_size, latent_dim)
        self.decoder = ae_decoder(image_size, latent_dim)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Return the list of metrics tracked during training / evaluation."""
        return [self.loss_tracker]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass: encode then decode.

        Args:
            x: Input image batch of shape ``(batch, H, W, 1)``.
            training: Whether to run in training mode.

        Returns:
            Reconstructed image batch of the same shape as *x*.
        """
        return self.decoder(self.encoder(x, training=training), training=training)

    def train_step(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single gradient-update step.

        Args:
            x: Batch of training images.

        Returns:
            Dictionary mapping metric names to their current values.
        """
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(x - self(x, training=True)))
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables)
        )
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single evaluation step (no gradient update).

        Args:
            x: Batch of validation images.

        Returns:
            Dictionary mapping metric names to their current values.
        """
        loss = tf.reduce_mean(tf.square(x - self(x, training=False)))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# ─────────────────────────────────────────────────────────────────────────────
# Variational Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class Sampling(layers.Layer):
    """Reparameterisation sampling layer: ``z = μ + ε · exp(0.5 · log σ²)``.

    Allows gradients to flow through the stochastic latent variable by
    separating the randomness (ε ~ N(0, I)) from the learned parameters
    (μ, log σ²).
    """

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Sample a latent vector using the reparameterisation trick.

        Args:
            inputs: Tuple of ``(z_mean, z_log_var)`` tensors, each of shape
                ``(batch, latent_dim)``.

        Returns:
            Sampled latent tensor of shape ``(batch, latent_dim)``.
        """
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + eps * tf.exp(0.5 * z_log_var)


def vae_encoder(image_size: int = 64, latent_dim: int = 64) -> keras.Model:
    """Build the VAE encoder: image → (z_mean, z_log_var, z_sample).

    Architecture mirrors *ae_encoder* but outputs three tensors —
    the distribution parameters and a reparameterised sample.

    Args:
        image_size: Spatial input resolution (square).
        latent_dim: Dimensionality of the latent distribution.

    Returns:
        A ``keras.Model`` with three outputs
        ``[z_mean, z_log_var, z_sample]``, each of shape ``(batch, latent_dim)``.
    """
    inp = keras.Input(shape=(image_size, image_size, 1))
    x = layers.Conv2D(32,  3, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z         = Sampling(name="z_sample")([z_mean, z_log_var])
    return keras.Model(inp, [z_mean, z_log_var, z], name="vae_encoder")


def vae_decoder(image_size: int = 64, latent_dim: int = 64) -> keras.Model:
    """Build the VAE decoder: latent sample → reconstructed image.

    Architecture mirrors *ae_decoder*.

    Args:
        image_size: Spatial output resolution (square).
        latent_dim: Dimensionality of the input latent vector.

    Returns:
        A ``keras.Model`` with output shape ``(batch, image_size, image_size, 1)``.
    """
    h = image_size // 8
    inp = keras.Input(shape=(latent_dim,))
    x = layers.Dense(h * h * 128, activation="relu")(inp)
    x = layers.Reshape((h, h, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32,  3, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv2D(1, 3, padding="same", activation="sigmoid", name="vae_recon")(x)
    return keras.Model(inp, out, name="vae_decoder")


class VAE(keras.Model):
    """β-VAE: Variational Autoencoder with optional KL weighting.

    Objective: ``E[reconstruction_loss] + β · KL(q(z|x) || p(z))``

    where the reconstruction loss is pixel-wise MSE summed over spatial
    dimensions and the KL term is computed analytically assuming
    ``q(z|x) = N(z_mean, exp(z_log_var))``.

    Attributes:
        encoder: Probabilistic encoder returning ``(z_mean, z_log_var, z)``.
        decoder: Decoder mapping latent samples to image reconstructions.
        beta: KL divergence weight (1.0 = standard VAE).
        latent_dim: Dimensionality of the latent space.
        total / recon / kl: Running-mean metrics tracked per step.
    """

    def __init__(
        self,
        image_size: int = 64,
        latent_dim: int = 64,
        beta: float = 1.0,
        **kwargs,
    ) -> None:
        """Initialise encoder, decoder, and loss-tracking metrics.

        Args:
            image_size: Spatial resolution of input images.
            latent_dim: Dimensionality of the latent space.
            beta: KL divergence weight (β-VAE coefficient).
            **kwargs: Forwarded to ``keras.Model.__init__``.
        """
        super().__init__(**kwargs)
        self.encoder    = vae_encoder(image_size, latent_dim)
        self.decoder    = vae_decoder(image_size, latent_dim)
        self.beta       = beta
        self.latent_dim = latent_dim
        self.total  = keras.metrics.Mean(name="total_loss")
        self.recon  = keras.metrics.Mean(name="recon_loss")
        self.kl     = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Return the list of metrics tracked during training / evaluation."""
        return [self.total, self.recon, self.kl]

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward pass: encode, sample, then decode.

        Args:
            x: Input image batch of shape ``(batch, H, W, 1)``.
            training: Whether to run in training mode.

        Returns:
            Tuple of ``(reconstruction, z_mean, z_log_var)``.
        """
        z_mean, z_log_var, z = self.encoder(x, training=training)
        return self.decoder(z, training=training), z_mean, z_log_var

    def _loss(
        self, x: tf.Tensor, training: bool
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute the total ELBO loss and its components.

        Args:
            x: Image batch.
            training: Training flag forwarded to sub-models.

        Returns:
            Tuple of ``(total_loss, reconstruction_loss, kl_loss)``.
        """
        z_mean, z_log_var, z = self.encoder(x, training=training)
        xhat  = self.decoder(z, training=training)
        recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - xhat), axis=[1, 2, 3]))
        kl    = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        return recon + self.beta * kl, recon, kl

    def train_step(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single gradient-update step.

        Args:
            x: Batch of training images.

        Returns:
            Dictionary mapping metric names to their current values.
        """
        with tf.GradientTape() as tape:
            total, recon, kl = self._loss(x, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(total, self.trainable_variables), self.trainable_variables)
        )
        self.total.update_state(total)
        self.recon.update_state(recon)
        self.kl.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single evaluation step (no gradient update).

        Args:
            x: Batch of validation images.

        Returns:
            Dictionary mapping metric names to their current values.
        """
        total, recon, kl = self._loss(x, training=False)
        self.total.update_state(total)
        self.recon.update_state(recon)
        self.kl.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def sample(self, n: int = 16) -> tf.Tensor:
        """Generate new images by sampling from the standard normal prior.

        Args:
            n: Number of images to generate.

        Returns:
            Image tensor of shape ``(n, image_size, image_size, 1)``.
        """
        z = tf.random.normal(shape=(n, self.latent_dim))
        return self.decoder(z, training=False)
