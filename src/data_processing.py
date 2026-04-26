"""Data pipeline utilities for the Medical-MNIST dataset.

Uses ``tf.data`` for efficient, reproducible input pipelines with
train / validation splitting.

Typical usage
-------------
>>> from src.data_processing import build_dataset
>>> train_ds, val_ds = build_dataset('AbdomenCT')
"""

# ── Standard library ──────────────────────────────────────────────────────────
import glob
import os
from typing import List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf


def _load_and_preprocess(path: tf.Tensor, image_size: int) -> tf.Tensor:
    """Decode an image file and normalise pixel values to [0, 1].

    Args:
        path: Scalar string tensor — the absolute file path.
        image_size: Target height and width in pixels (square resize).

    Returns:
        A float32 tensor of shape ``(image_size, image_size, 1)``.
    """
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=1, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([image_size, image_size, 1])
    return img


def build_dataset(
    region: str,
    data_root: str,
    image_size: int = 64,
    batch_size: int = 32,
    val_split: float = 0.15,
    shuffle_buf: int = 1000,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build batched train and validation ``tf.data.Dataset`` objects for one region.

    Images are loaded from ``<data_root>/<region>/`` and split
    deterministically into train and validation subsets.

    Args:
        region: Subfolder name under *data_root* (e.g. ``'AbdomenCT'``).
        data_root: Root directory containing one subfolder per region.
        image_size: Spatial resolution to resize images to (square).
        batch_size: Number of samples per batch.
        val_split: Fraction of images reserved for validation.
        shuffle_buf: Buffer size used by ``tf.data.Dataset.shuffle``.
        seed: Random seed for the train / val split.

    Returns:
        A ``(train_ds, val_ds)`` tuple of prefetched, batched datasets.

    Raises:
        FileNotFoundError: If *region* folder is missing or contains no images.
    """
    region_dir = os.path.join(data_root, region)
    if not os.path.isdir(region_dir):
        raise FileNotFoundError(f"Region folder not found: {region_dir}")

    all_files: List[str] = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
        all_files.extend(glob.glob(os.path.join(region_dir, ext)))
    paths = sorted(set(all_files))
    if not paths:
        raise FileNotFoundError(f"No images found in {region_dir}")

    print(f"{region}: {len(paths)} images found")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(paths))
    n_val = max(1, int(len(paths) * val_split))
    val_paths = [paths[i] for i in shuffled[:n_val]]
    trn_paths = [paths[i] for i in shuffled[n_val:]]

    def _make(path_list: List[str], do_shuffle: bool) -> tf.data.Dataset:
        """Create a batched, prefetched dataset from a list of file paths."""
        ds = tf.data.Dataset.from_tensor_slices(path_list)
        ds = ds.map(
            lambda p: _load_and_preprocess(p, image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if do_shuffle:
            ds = ds.shuffle(shuffle_buf, seed=seed, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return _make(trn_paths, True), _make(val_paths, False)
