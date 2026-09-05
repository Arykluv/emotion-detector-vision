"""Prepare the emotion classification dataset from FER-2013 (Phase 4).

Reads the original fer2013.csv, keeps all seven emotions (angry, disgust,
fear, happy, sad, surprise, neutral), splits them into train and test sets,
and writes:

  - 48x48 grayscale PNG images per class (easy to inspect)
  - normalized NumPy arrays (ready for the training pipeline)

No neural network is built or trained here - that happens in a later phase.
The original CSV file is never modified.
"""

import argparse
import csv
import os
import shutil
import zipfile
from urllib.request import Request, urlopen

import cv2
import numpy as np

# Alternate hosts of the original FER-2013 dataset (tried in order).
# The first is the full canonical CSV (emotion, pixels, Usage) packed in a zip.
FER2013_SOURCES = [
    "https://huggingface.co/datasets/chitradrishti/fer2013/resolve/main/fer2013.csv.zip",
    "https://storage.googleapis.com/mledu-datasets/fer2013.csv",
]

# FER-2013 emotion ids -> class names (all seven emotions)
CLASS_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}
USE_LABELS = sorted(CLASS_NAMES)  # [0, 1, 2, 3, 4, 5, 6]

# Map each kept label to a contiguous class index so the CNN outputs match.
# With all seven emotions kept, this is the identity mapping.
LABEL_TO_INDEX = {label: index for index, label in enumerate(USE_LABELS)}

IMAGE_SIZE = 48  # FER-2013 images are 48x48 grayscale


def download_file(url, dest_path):
    """Stream a single URL to dest_path and return the number of bytes."""
    # Some hosts reject the default Python user agent
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response, open(dest_path, "wb") as dest:
        total = 0
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            dest.write(chunk)
            total += len(chunk)
    return total


def download_fer2013(dest_path):
    """Download the original FER-2013 CSV if it does not exist yet.

    Handles plain CSV files and CSV packed in a zip archive.
    Returns True on success, False if every source fails.
    """
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"Using existing CSV: {dest_path}")
        return True

    tmp_path = dest_path + ".tmp"
    for url in FER2013_SOURCES:
        print(f"Trying to download from: {url}")
        try:
            total = download_file(url, tmp_path)

            if url.endswith(".zip"):
                # Extract the first CSV found inside the zip archive
                with zipfile.ZipFile(tmp_path) as archive:
                    csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                    if not csv_names:
                        raise RuntimeError("zip archive contains no CSV file")
                    with archive.open(csv_names[0]) as src, open(dest_path, "wb") as dest:
                        dest.write(src.read())
                os.remove(tmp_path)
            else:
                os.replace(tmp_path, dest_path)

            print(f"Downloaded to: {dest_path} ({total / 1_000_000:.1f} MB)")
            return True
        except Exception as exc:
            print(f"  failed: {exc.__class__.__name__}: {exc}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print("Could not download FER-2013 from any source.")
    print(f"Please download fer2013.csv yourself and save it as: {dest_path}")
    return False


def read_rows(csv_path):
    """Read the CSV and return (label, pixel_values, split) for kept rows."""
    rows = []
    with open(csv_path, newline="") as source:
        # The header may contain spaces (e.g. "emotion, Usage, pixels"),
        # so strip whitespace from every column name
        raw_reader = csv.reader(source)
        headers = [name.strip() for name in next(raw_reader)]
        reader = csv.DictReader(source, fieldnames=headers)

        for row in reader:
            label = int(row["emotion"])
            if label not in USE_LABELS:
                continue  # skip emotions we do not need

            pixels = [int(p) for p in row["pixels"].split()]
            if len(pixels) != IMAGE_SIZE * IMAGE_SIZE:
                continue  # skip malformed rows (should not happen)

            # PublicTest and PrivateTest are both used as the test set
            split = "train" if row["Usage"] == "Training" else "test"

            rows.append((label, pixels, split))
    return rows


def save_images(rows, output_dir):
    """Write one grayscale 48x48 PNG per row into train/ and test/ folders."""
    counts = {"train": {}, "test": {}}
    for index, (label, pixels, split) in enumerate(rows):
        class_name = CLASS_NAMES[label]
        folder = os.path.join(output_dir, split, class_name)
        os.makedirs(folder, exist_ok=True)
        counts[split][class_name] = counts[split].get(class_name, 0) + 1

        # FER-2013 pixels are already 48x48 grayscale, reshape keeps the size
        image = np.array(pixels, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE)
        cv2.imwrite(os.path.join(folder, f"{index:06d}.png"), image)

        if (index + 1) % 5000 == 0:
            print(f"  saved {index + 1}/{len(rows)} images")
    return counts


def save_arrays(rows, output_dir):
    """Save normalized float arrays and labels for the training pipeline."""
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for label, pixels, split in rows:
        # Convert to float and normalize pixel values to the range [0, 1]
        image = np.array(pixels, dtype=np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE)
        image = image / 255.0

        # Store the contiguous class index, not the raw FER-2013 id
        class_index = LABEL_TO_INDEX[label]

        if split == "train":
            train_images.append(image)
            train_labels.append(class_index)
        else:
            test_images.append(image)
            test_labels.append(class_index)

    np.save(os.path.join(output_dir, "X_train.npy"), np.stack(train_images))
    np.save(os.path.join(output_dir, "y_train.npy"), np.asarray(train_labels, dtype=np.int32))
    np.save(os.path.join(output_dir, "X_test.npy"), np.stack(test_images))
    np.save(os.path.join(output_dir, "y_test.npy"), np.asarray(test_labels, dtype=np.int32))
    np.save(
        os.path.join(output_dir, "class_names.npy"),
        np.array([CLASS_NAMES[label] for label in USE_LABELS]),
    )


def print_summary(counts):
    """Print the dataset statistics required by Phase 4."""
    labels = [CLASS_NAMES[label] for label in USE_LABELS]

    print("=" * 45)
    print("Dataset summary")
    print("=" * 45)
    print(f"Number of classes            : {len(labels)}")
    print(f"Image dimensions             : {IMAGE_SIZE}x{IMAGE_SIZE} grayscale")
    print()
    print("Training images per class:")
    for name in labels:
        print(f"  {name:7s}: {counts['train'].get(name, 0)}")
    print()
    print("Test images per class:")
    for name in labels:
        print(f"  {name:7s}: {counts['test'].get(name, 0)}")
    print()
    print(f"Total training images        : {sum(counts['train'].values())}")
    print(f"Total test images            : {sum(counts['test'].values())}")
    print("=" * 45)


def main():
    parser = argparse.ArgumentParser(description="Prepare the FER-2013 dataset for training.")
    parser.add_argument(
        "--source",
        default="dataset/fer2013.csv",
        help="Path to the original fer2013.csv (downloaded if missing).",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/prepared",
        help="Where the prepared images and arrays are written.",
    )
    args = parser.parse_args()

    if not download_fer2013(args.source):
        raise SystemExit(1)

    # Start fresh: stale per-class PNGs from an earlier run (e.g. a 3-class
    # subset) must not survive into the new layout.
    for split in ("train", "test"):
        split_dir = os.path.join(args.output_dir, split)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)

    print("Reading and filtering FER-2013 rows (all seven emotions)...")
    rows = read_rows(args.source)
    print(f"Kept {len(rows)} rows.")

    print("Saving PNG images...")
    counts = save_images(rows, args.output_dir)

    print("Saving normalized NumPy arrays...")
    save_arrays(rows, args.output_dir)

    print()
    print_summary(counts)
    print(f"\nPrepared dataset written to: {args.output_dir}")


if __name__ == "__main__":
    main()