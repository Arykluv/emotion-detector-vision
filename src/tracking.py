"""Multi-face tracking with stable per-person IDs (Phase 7).

Haar detects faces fresh every frame, so boxes jitter and flicker between
people. ``FaceTracker`` matches detections to existing tracks with IoU and
keeps a label per track: labels therefore follow the same person, survive a
few frames of occlusion, and never swap between two nearby faces.
"""

from collections import OrderedDict


class TrackedFace:
    """One person being followed, with their classification state."""

    __slots__ = ("track_id", "box", "misses", "label", "confidence", "streak")

    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = box
        self.misses = 0
        self.label = None  # displayed label after confidence gating
        self.confidence = None
        self.streak = 0  # how many classify cycles this label has been stable


def iou(box_a, box_b):
    """Intersection-over-union of two (x, y, width, height) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """Assign stable ids to faces using greedy IoU matching."""

    def __init__(self, iou_threshold=0.3, max_misses=6, max_tracks=10):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.max_tracks = max_tracks
        self.tracks = OrderedDict()  # track_id -> TrackedFace
        self.next_id = 0

    def update(self, boxes, labels=None):
        """Match new detections to existing tracks and advance them.

        ``boxes`` is the list of (x, y, w, h) rectangles from the detector.
        ``labels`` (optional, aligned with ``boxes``) is a list whose entries
        are either None or ``(display_label, confidence)`` from the latest
        classification cycle. A track only counts a label as confirmed once it
        has repeated for several cycles (streak), suppressing flicker.

        Returns a list of ``(track_id, box, label, confidence, streak)`` for
        the tracks present this frame (surviving tracks keep their last box).
        """
        boxes = list(boxes)
        labels = list(labels) if labels is not None else [None] * len(boxes)

        matched = set()
        for det_index, box in enumerate(boxes):
            best_id, best_iou = None, self.iou_threshold
            for track_id, track in self.tracks.items():
                if track_id in matched:
                    continue
                value = iou(box, track.box)
                if value > best_iou:
                    best_id, best_iou = track_id, value
            if best_id is None:
                track = self._add_track(box, labels[det_index])
                matched.add(track.track_id)  # new tracks are not "missing"
            else:
                track = self.tracks[best_id]
                track.box = box
                track.misses = 0
                self._apply_label(track, labels[det_index])
                matched.add(best_id)

        # Tracks that were not matched this frame are allowed to age; their
        # last-known box keeps the label alive through brief occlusions.
        for track in list(self.tracks.values()):
            if track.track_id not in matched:
                track.misses += 1
                if track.misses > self.max_misses:
                    del self.tracks[track.track_id]

        # Enforce the track cap (drop the oldest)
        while len(self.tracks) > self.max_tracks:
            self.tracks.popitem(last=False)

        return [
            (t.track_id, t.box, t.label, t.confidence, t.streak)
            for t in self.tracks.values()
        ]

    def _add_track(self, box, label):
        track = TrackedFace(self.next_id, box)
        self._apply_label(track, label)
        self.tracks[self.next_id] = track
        self.next_id += 1
        return track

    @staticmethod
    def _apply_label(track, label):
        """Update a track's label; None leaves the previous label untouched."""
        if label is None:
            return
        display, confidence = label
        if track.label == display:
            track.streak += 1
        else:
            track.streak = 1
        track.label = display
        track.confidence = confidence