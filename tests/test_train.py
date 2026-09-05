"""Tests for the training pipeline helpers (train_model.py)."""

import numpy as np
import pytest
import torch

from train_model import (
    EmotionDataset,
    load_arrays,
    make_splits,
    remap_labels,
)


def test_make_splits_respects_ratio_and_seed():
    rng = np.random.RandomState(0)
    x = rng.rand(100, 48, 48).astype(np.float32)
    y = rng.randint(0, 7, size=100)

    a = make_splits(x, y, val_ratio=0.2, seed=42)
    b = make_splits(x, y, val_ratio=0.2, seed=42)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)

    x_tr, y_tr, x_val, y_val = a
    assert x_tr.shape[0] == y_tr.shape[0] == 80
    assert x_val.shape[0] == y_val.shape[0] == 20

    union = np.concatenate([x_tr, x_val])
    for row in x:
        assert np.any(np.all(union == row, axis=(1, 2)))


def test_remap_labels_produces_contiguous_indices():
    out = remap_labels([5, 5, 0, 3, 0])
    # sorted unique classes {0, 3, 5} -> 0->0, 3->1, 5->2
    np.testing.assert_array_equal(out, [2, 2, 0, 1, 0])
    assert out.dtype == np.int64


def test_remap_labels_handles_numpy_input():
    out = remap_labels(np.array([7, 7, 2, 2, 2]))
    # sorted unique {2, 7} -> 2->0, 7->1
    np.testing.assert_array_equal(out, [1, 1, 0, 0, 0])


def test_load_arrays_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_arrays(str(tmp_path), ["X_train.npy"])


def test_load_arrays_optional_returns_none(tmp_path):
    assert load_arrays(str(tmp_path), ["missing.npy"], required=False) is None


def test_emotion_dataset_shapes(tmp_path):
    x = np.zeros((4, 48, 48), dtype=np.float32)
    y = np.zeros(4, dtype=np.int64)
    dataset = EmotionDataset(x, y)
    assert len(dataset) == 4
    image, label = dataset[0]
    assert tuple(image.shape) == (1, 48, 48)
    assert image.dtype == torch.float32
    assert label.item() == 0


def test_augmentation_is_reproducible_with_seed():
    x = np.ones((1, 48, 48), dtype=np.float32)
    y = np.zeros(1, dtype=np.int64)
    dataset = EmotionDataset(x, y, augment=True)

    torch.manual_seed(7)
    first, _ = dataset[0]
    torch.manual_seed(7)
    second, _ = dataset[0]
    torch.equal(first, second)


def test_augmentation_keeps_image_shape():
    x = np.zeros((1, 48, 48), dtype=np.float32)
    y = np.zeros(1, dtype=np.int64)
    dataset = EmotionDataset(x, y, augment=True)
    torch.manual_seed(3)
    image, _ = dataset[0]
    assert tuple(image.shape) == (1, 48, 48)
    assert torch.isfinite(image).all()