"""Tests for the wave-to-close gesture detector (src/gesture.py)."""

import numpy as np

from src.gesture import WaveDetector


def test_no_detection_on_static_camera():
    detector = WaveDetector()
    frame = np.full((96, 128), 128, dtype=np.uint8)
    for _ in range(80):
        assert not detector.update(frame)


def test_no_detection_on_slow_head_sway():
    # A tiny 1px whole-frame sway is below the motion threshold.
    detector = WaveDetector()
    background = np.zeros((96, 128), dtype=np.uint8)
    for t in range(80):
        shift = 1 if (t // 5) % 2 == 0 else -1
        frame = np.roll(background, shift, axis=1)[0:96, 0:128]
        assert not detector.update(frame)


def test_detects_waving_hand():
    detector = WaveDetector()
    detected = False
    # A bright block oscillates smoothly left/right (triangle wave),
    # i.e. continuous motion that keeps reversing direction.
    sequence = [0, 4, 8, 4, 0, -4, -8, -4]  # px offset, reverses every 4 steps
    for t in range(120):
        frame = np.zeros((96, 128), dtype=np.uint8)
        x = 40 + sequence[t % len(sequence)]
        frame[40:60, x : x + 24] = 255
        if detector.update(frame):
            detected = True
            break
    assert detected


def test_detector_recovers_after_firing():
    detector = WaveDetector()
    fired_at = None
    sequence = [0, 4, 8, 4, 0, -4, -8, -4]
    for t in range(120):
        frame = np.zeros((96, 128), dtype=np.uint8)
        x = 40 + sequence[t % len(sequence)]
        frame[40:60, x : x + 24] = 255
        if detector.update(frame):
            fired_at = t
            break
    assert fired_at is not None
    # After a detection the history is cleared; static frames stay quiet.
    frame = np.full((96, 128), 128, dtype=np.uint8)
    for _ in range(20):
        assert not detector.update(frame)