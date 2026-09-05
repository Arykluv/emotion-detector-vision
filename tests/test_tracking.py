"""Tests for the multi-face tracker (src/tracking.py)."""

import pytest

from src.tracking import FaceTracker, iou


def test_iou_identical_boxes():
    assert iou((20, 20, 100, 100), (20, 20, 100, 100)) == 1.0


def test_iou_disjoint_boxes():
    assert iou((0, 0, 50, 50), (200, 200, 50, 50)) == 0.0


def test_iou_partial_overlap():
    # 50x100 overlap over a 150x100 union -> 5000 / 15000
    assert iou((0, 0, 100, 100), (50, 0, 100, 100)) == pytest.approx(1 / 3)


def test_first_observations_get_distinct_ids():
    tracker = FaceTracker()
    tracks = tracker.update([(0, 0, 100, 100), (200, 200, 100, 100)])
    ids = [t[0] for t in tracks]
    assert len(set(ids)) == 2


def test_small_motion_keeps_ids():
    tracker = FaceTracker()
    tracker.update([(0, 0, 100, 100)])
    tracks = tracker.update([(4, 4, 98, 98)], labels=[("happy", 0.9)])
    assert tracks[0][0] == 0
    assert tracks[0][2] == "happy"


def test_label_follows_face_regardless_of_detector_order():
    # Person A (left, happy) and person B (right, sad). Haar frequently
    # returns faces in a different order each frame; per-track labels must
    # follow the PERSON, not the list position.
    tracker = FaceTracker()
    tracker.update([(0, 0, 100, 100), (220, 220, 100, 100)],
                   labels=[("happy", 0.9), ("sad", 0.8)])
    tracker.update([(224, 224, 100, 100), (4, 4, 100, 100)],
                   labels=[("sad", 0.8), ("happy", 0.9)])
    by_id = {t[0]: t[2] for t in tracker.update([(4, 4, 100, 100), (224, 224, 100, 100)])}
    assert by_id == {0: "happy", 1: "sad"}


def test_streak_grows_slowly():
    tracker = FaceTracker()
    tracker.update([(0, 0, 100, 100)], labels=[("happy", 0.9)])
    tracker.update([(0, 0, 100, 100)], labels=[("happy", 0.9)])
    tracks = tracker.update([(0, 0, 100, 100)], labels=[("happy", 0.9)])
    assert tracks[0][4] == 3  # streak resets each cycle, confirmed at 3


def test_track_dies_after_too_many_misses():
    tracker = FaceTracker(max_misses=3)
    tracker.update([(0, 0, 100, 100)])
    for _ in range(3):  # misses 1..3 -> never over the limit, still alive
        tracks = tracker.update([])
        assert len(tracks) == 1
    tracks = tracker.update([])  # misses == 4 -> dropped
    assert tracks == []


def test_none_label_leaves_previous_intact():
    tracker = FaceTracker()
    tracker.update([(0, 0, 100, 100)], labels=[("happy", 0.9)])
    tracks = tracker.update([(0, 0, 100, 100)], labels=[None])
    assert tracks[0][2] == "happy"
    assert tracks[0][4] == 1  # unchanged display -> streak stays


def test_identical_track_id_is_the_same_person():
    # Reconstructed after the aging bug was fixed: creating a track must not
    # count as a "miss" (that would silently shorten its lifetime by one).
    tracker = FaceTracker(max_misses=3)
    first = tracker.update([(0, 0, 100, 100)])
    assert first[0][0] == 0
    assert first[0][1] == (0, 0, 100, 100)


def test_track_cap_drops_oldest():
    tracker = FaceTracker(max_tracks=2)
    tracker.update([(0, 0, 50, 50)])
    tracker.update([(500, 500, 50, 50)])     # far away -> a new person
    tracker.update([(1000, 1000, 50, 50)])   # third person, cap hits
    assert {t.track_id for t in tracker.tracks.values()} == {1, 2}
    ids = [t[0] for t in tracker.update([])]
    assert len(ids) == 2
    assert 0 not in ids  # the oldest track was evicted