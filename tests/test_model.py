"""Unit tests for src/model.py (Autoencoder and VAE)."""

# ── Standard library ──────────────────────────────────────────────────────────
import pytest

# ── Third-party ───────────────────────────────────────────────────────────────
import tensorflow as tf

# ── Local ─────────────────────────────────────────────────────────────────────
from src.model import Autoencoder, VAE, ae_decoder, ae_encoder, vae_decoder, vae_encoder

IMAGE_SIZE = 32   # small size for fast tests
LATENT_DIM = 16
BATCH      = 4


def _fake_batch(image_size: int = IMAGE_SIZE, batch: int = BATCH) -> tf.Tensor:
    """Return a random float32 image batch in [0, 1]."""
    return tf.random.uniform((batch, image_size, image_size, 1))


# ─────────────────────────────────────────────────────────────────────────────
class TestAEEncoder:
    def test_output_shape(self):
        enc = ae_encoder(IMAGE_SIZE, LATENT_DIM)
        out = enc(_fake_batch())
        assert out.shape == (BATCH, LATENT_DIM)


class TestAEDecoder:
    def test_output_shape(self):
        dec = ae_decoder(IMAGE_SIZE, LATENT_DIM)
        z   = tf.random.normal((BATCH, LATENT_DIM))
        out = dec(z)
        assert out.shape == (BATCH, IMAGE_SIZE, IMAGE_SIZE, 1)

    def test_output_range(self):
        dec = ae_decoder(IMAGE_SIZE, LATENT_DIM)
        z   = tf.random.normal((BATCH, LATENT_DIM))
        out = dec(z)
        assert float(tf.reduce_min(out)) >= 0.0
        assert float(tf.reduce_max(out)) <= 1.0


class TestAutoencoder:
    def test_forward_output_shape(self):
        ae  = Autoencoder(IMAGE_SIZE, LATENT_DIM)
        out = ae(_fake_batch())
        assert out.shape == (BATCH, IMAGE_SIZE, IMAGE_SIZE, 1)

    def test_train_step_returns_loss(self):
        ae = Autoencoder(IMAGE_SIZE, LATENT_DIM)
        ae.compile(optimizer="adam")
        for b in tf.data.Dataset.from_tensors(_fake_batch()).batch(BATCH).take(1):
            result = ae.train_step(b)
        assert "loss" in result
        assert float(result["loss"]) >= 0.0

    def test_metrics_property(self):
        ae = Autoencoder(IMAGE_SIZE, LATENT_DIM)
        assert len(ae.metrics) == 1
        assert ae.metrics[0].name == "loss"


# ─────────────────────────────────────────────────────────────────────────────
class TestVAEEncoder:
    def test_output_shapes(self):
        enc = vae_encoder(IMAGE_SIZE, LATENT_DIM)
        z_mean, z_log_var, z = enc(_fake_batch())
        for t in (z_mean, z_log_var, z):
            assert t.shape == (BATCH, LATENT_DIM)


class TestVAEDecoder:
    def test_output_shape(self):
        dec = vae_decoder(IMAGE_SIZE, LATENT_DIM)
        z   = tf.random.normal((BATCH, LATENT_DIM))
        out = dec(z)
        assert out.shape == (BATCH, IMAGE_SIZE, IMAGE_SIZE, 1)


class TestVAE:
    def test_forward_output_shapes(self):
        vae = VAE(IMAGE_SIZE, LATENT_DIM)
        recon, z_mean, z_log_var = vae(_fake_batch())
        assert recon.shape      == (BATCH, IMAGE_SIZE, IMAGE_SIZE, 1)
        assert z_mean.shape     == (BATCH, LATENT_DIM)
        assert z_log_var.shape  == (BATCH, LATENT_DIM)

    def test_train_step_returns_all_losses(self):
        vae = VAE(IMAGE_SIZE, LATENT_DIM)
        vae.compile(optimizer="adam")
        for b in tf.data.Dataset.from_tensors(_fake_batch()).batch(BATCH).take(1):
            result = vae.train_step(b)
        for key in ("total_loss", "recon_loss", "kl_loss"):
            assert key in result

    def test_sample_shape(self):
        vae = VAE(IMAGE_SIZE, LATENT_DIM)
        vae(_fake_batch())  # build weights
        samples = vae.sample(n=8)
        assert samples.shape == (8, IMAGE_SIZE, IMAGE_SIZE, 1)

    def test_sample_range(self):
        vae = VAE(IMAGE_SIZE, LATENT_DIM)
        vae(_fake_batch())
        samples = vae.sample(n=4)
        assert float(tf.reduce_min(samples)) >= 0.0
        assert float(tf.reduce_max(samples)) <= 1.0

    def test_metrics_property(self):
        vae = VAE(IMAGE_SIZE, LATENT_DIM)
        names = [m.name for m in vae.metrics]
        assert names == ["total_loss", "recon_loss", "kl_loss"]
