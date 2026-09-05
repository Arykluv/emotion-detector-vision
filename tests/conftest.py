"""Shared fixtures for the emotion-detector test suite."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_SIZE = 48


@pytest.fixture
def csv_builder(tmp_path):
    """Build tiny FER-2013-style CSVs (label, pixels, Usage rows)."""

    def build(rows, pixel_count=IMAGE_SIZE * IMAGE_SIZE):
        path = tmp_path / "fer.csv"
        with open(path, "w", encoding="utf-8") as dest:
            dest.write("emotion,pixels,Usage\n")
            for label, usage in rows:
                pixels = " ".join(["128"] * pixel_count)
                dest.write(f"{label},{pixels},{usage}\n")
        return str(path)

    return build


@pytest.fixture
def tiny_checkpoint(tmp_path, tmp_path_factory):
    """Train a small scratch CNN and save it as a checkpoint dict."""
    import torch

    from src.emotion_model import build_model

    model = build_model(num_classes=7, architecture="cnn", pretrained=False)
    model_path = tmp_path / "tiny_cnn.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": ["angry", "disgust", "fear", "happy",
                            "sad", "surprise", "neutral"],
            "architecture": "cnn",
        },
        model_path,
    )
    return str(model_path)