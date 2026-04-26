"""Unit tests for src/data_processing.py."""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import tempfile

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

# ── Local ─────────────────────────────────────────────────────────────────────
from src.data_processing import build_dataset


def _make_fake_region(root: str, region: str, n: int = 20) -> str:
    """Write *n* random 64×64 grayscale PNG files under ``root/region/``."""
    region_dir = os.path.join(root, region)
    os.makedirs(region_dir, exist_ok=True)
    for i in range(n):
        arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(os.path.join(region_dir, f"img_{i:04d}.png"))
    return region_dir


class TestBuildDataset:
    """Tests for ``build_dataset``."""

    def test_returns_two_datasets(self, tmp_path):
        _make_fake_region(str(tmp_path), "TestRegion", n=20)
        train_ds, val_ds = build_dataset(
            "TestRegion", data_root=str(tmp_path), image_size=64, batch_size=4
        )
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(val_ds,   tf.data.Dataset)

    def test_batch_shape(self, tmp_path):
        _make_fake_region(str(tmp_path), "TestRegion", n=20)
        train_ds, _ = build_dataset(
            "TestRegion", data_root=str(tmp_path), image_size=32, batch_size=4
        )
        for batch in train_ds.take(1):
            assert batch.shape == (4, 32, 32, 1)

    def test_pixel_range(self, tmp_path):
        _make_fake_region(str(tmp_path), "TestRegion", n=20)
        train_ds, _ = build_dataset(
            "TestRegion", data_root=str(tmp_path), image_size=32, batch_size=8
        )
        for batch in train_ds.take(1):
            assert float(tf.reduce_min(batch)) >= 0.0
            assert float(tf.reduce_max(batch)) <= 1.0

    def test_missing_region_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_dataset("DoesNotExist", data_root=str(tmp_path))

    def test_val_split_proportion(self, tmp_path):
        _make_fake_region(str(tmp_path), "TestRegion", n=100)
        train_ds, val_ds = build_dataset(
            "TestRegion",
            data_root=str(tmp_path),
            image_size=32,
            batch_size=1,
            val_split=0.2,
        )
        n_train = sum(1 for _ in train_ds)
        n_val   = sum(1 for _ in val_ds)
        assert n_val == 20
        assert n_train == 80
