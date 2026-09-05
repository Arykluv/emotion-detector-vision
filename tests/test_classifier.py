"""Tests for the runtime classifier (src/emotion_classifier.py)."""

import numpy as np

from src.emotion_classifier import EmotionClassifier, INPUT_SIZE


def test_classifier_loads_checkpoint_and_names(tiny_checkpoint):
    classifier = EmotionClassifier(tiny_checkpoint)
    assert len(classifier.class_names) == 7


def test_predict_batch_returns_valid_distribution(tiny_checkpoint):
    classifier = EmotionClassifier(tiny_checkpoint)
    crops = [np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)] * 3
    results = classifier.predict_batch(crops)
    assert len(results) == 3
    for label, confidence in results:
        assert label in classifier.class_names
        assert 0.0 <= confidence <= 1.0


def test_predict_batch_detailed_reports_second_class(tiny_checkpoint):
    classifier = EmotionClassifier(tiny_checkpoint)
    crops = [np.ones((48, 48), dtype=np.uint8)]
    label, confidence, second, second_conf = classifier.predict_batch(
        crops, detailed=True
    )[0]
    assert label in classifier.class_names
    assert second in classifier.class_names
    assert confidence >= second_conf


def test_predict_batch_empty_input(tiny_checkpoint):
    classifier = EmotionClassifier(tiny_checkpoint)
    assert classifier.predict_batch([]) == []


def test_predict_single_face(tiny_checkpoint):
    classifier = EmotionClassifier(tiny_checkpoint)
    label, confidence = classifier.predict(np.zeros((48, 48), dtype=np.uint8))
    assert label in classifier.class_names
    assert isinstance(label, str)