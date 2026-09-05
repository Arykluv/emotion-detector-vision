"""Tests for pure helpers in the live app (main.py)."""

import re

import cv2
import numpy as np

import main as app
from main import mood_bucket, stable_label, timestamp


def test_stable_label_rejects_low_confidence():
    assert stable_label("happy", 0.20, "sad", 0.19) is None


def test_stable_label_marks_tied_top_two():
    assert stable_label("angry", 0.45, "sad", 0.33) == "angry/sad"


def test_stable_label_accepts_decisive_prediction():
    assert stable_label("happy", 0.72, "sad", 0.11) == "happy"


def test_mood_bucket_splits_ambiguous_labels():
    assert mood_bucket("happy/sad") == "?"
    assert mood_bucket("neutral") == "neutral"
    assert mood_bucket("angry/") == "?"


def test_timestamp_is_compact_and_local():
    assert re.fullmatch(r"\d{8}_\d{6}", timestamp())


def test_mood_panel_draws_on_any_frame():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    app.draw_mood_panel(frame, 10, 10, {"happy": 3, "sad": 1})
    assert frame.any()  # the bars actually painted something


def test_emotion_label_smoke():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    app.draw_emotion_label(frame, 20, 60, 80, "happy", 0.9)
    assert frame.any()


def test_parse_args_defaults(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys, "argv", ["main.py"])
    args = app.parse_args()
    assert args.camera == 0
    assert args.tta is True
    assert args.min_neighbors == 5