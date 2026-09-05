"""Tests for the dataset preparation pipeline (prepare_dataset.py)."""

import numpy as np

from prepare_dataset import (
    CLASS_NAMES,
    IMAGE_SIZE,
    LABEL_TO_INDEX,
    read_rows,
    save_arrays,
    save_images,
)
from tests.conftest import IMAGE_SIZE as TEST_IMAGE_SIZE


def test_all_seven_emotions_kept():
    assert len(CLASS_NAMES) == 7
    names = [CLASS_NAMES[label] for label in sorted(CLASS_NAMES)]
    assert names == ["angry", "disgust", "fear", "happy", "sad",
                     "surprise", "neutral"]


def test_label_mapping_is_identity_for_seven_classes():
    expected = {label: index for index, label in enumerate(sorted(CLASS_NAMES))}
    assert LABEL_TO_INDEX == expected


def test_image_size_matches_48x48():
    assert IMAGE_SIZE == TEST_IMAGE_SIZE == 48


def test_read_rows_keeps_train_and_test_splits(csv_builder):
    path = csv_builder([
        (3, "Training"),
        (6, "PrivateTest"),
        (0, "PublicTest"),
    ])
    rows = read_rows(path)
    splits = [split for _, _, split in rows]
    assert splits == ["train", "test", "test"]
    assert rows[0][0] == 3
    assert len(rows[0][1]) == IMAGE_SIZE * IMAGE_SIZE


def test_read_rows_skips_excluded_emotions(csv_builder):
    # emotion id 99 is not one of the seven -> dropped
    path = csv_builder([(99, "Training"), (1, "Training")])
    rows = read_rows(path)
    assert [label for label, _, _ in rows] == [1]


def test_read_rows_skips_malformed_pixel_rows(csv_builder):
    path = csv_builder([(3, "Training")], pixel_count=100)
    assert read_rows(path) == []


def test_save_arrays_write_normalized_images(csv_builder, tmp_path):
    path = csv_builder([
        (0, "Training"),
        (1, "Training"),
        (6, "PrivateTest"),
    ])
    rows = read_rows(path)
    save_arrays(rows, str(tmp_path))

    x_train = np.load(tmp_path / "X_train.npy")
    y_train = np.load(tmp_path / "y_train.npy")
    assert x_train.shape == (2, 48, 48)
    assert x_train.dtype == np.float32
    assert x_train.min() >= 0.0 and x_train.max() <= 1.0
    assert y_train.tolist() == [0, 1]

    x_test = np.load(tmp_path / "X_test.npy")
    y_test = np.load(tmp_path / "y_test.npy")
    assert x_test.shape == (1, 48, 48)
    assert y_test.tolist() == [6]

    names = np.load(tmp_path / "class_names.npy").tolist()
    assert names == ["angry", "disgust", "fear", "happy", "sad",
                     "surprise", "neutral"]


def test_save_images_creates_per_class_folders(csv_builder, tmp_path):
    path = csv_builder([(3, "Training")])
    rows = read_rows(path)
    counts = save_images(rows, str(tmp_path))

    train_dir = tmp_path / "train" / "happy"
    assert train_dir.is_dir()
    assert len(list(train_dir.glob("*.png"))) == 1
    assert counts["train"].get("happy") == 1