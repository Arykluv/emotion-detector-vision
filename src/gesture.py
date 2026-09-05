"""Hand-wave detection for "wave to close" (Phase 7).

A wave is periodic horizontal motion: the hand moves right, then left, then
right... We detect it with dense optical flow (Farneback) on a downscaled
grayscale stream and count how often the dominant horizontal flow direction
flips within a short rolling window. A stable face does not move, so it
produces (almost) no flow and cannot trigger the gesture by itself.

Everything is classic OpenCV + numpy: no extra dependencies beyond the
project's pinned `opencv-python`.
"""

import cv2
import numpy as np
from collections import deque


class WaveDetector:
    """Feed it grayscale frames; it returns True once a wave is observed."""

    def __init__(self, window=45, flips_required=5, motion_threshold=1.2,
                 min_motion_frac=0.002, active_ratio=0.4):
        self.window = window            # flow samples examined (1.5 s @ 30 fps)
        self.flips_required = flips_required
        self.motion_threshold = motion_threshold   # |flow| px/frame to count
        self.min_motion_frac = min_motion_frac     # frame fraction for "moving"
        self.active_ratio = active_ratio           # share of window that moves
        self._flow_u = deque(maxlen=window)        # dominant horizontal flow
        self._active = deque(maxlen=window)        # is this frame "moving"?
        self._previous = None

    def reset(self):
        """Drop all history (done after a detection to avoid repeats)."""
        self._flow_u.clear()
        self._active.clear()
        self._previous = None

    def update(self, gray):
        """Feed one grayscale frame (uint8 HxW); True fires once on a wave."""
        if self._previous is None:
            self._previous = gray
            return False

        flow = cv2.calcOpticalFlowFarneback(
            self._previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self._previous = gray

        moving = np.hypot(flow[..., 0], flow[..., 1]) > self.motion_threshold
        active = float(moving.mean()) > self.min_motion_frac
        # Dominant horizontal speed across the moving pixels (0 when still)
        if active:
            u = float(flow[moving, 0].mean())
        else:
            u = 0.0
        self._flow_u.append(u if abs(u) > self.motion_threshold else 0.0)
        self._active.append(active)

        if self._is_wave():
            self.reset()
            return True
        return False

    def _is_wave(self):
        """True when the window shows steady, independently-flipping motion."""
        if len(self._flow_u) < self.window:
            return False
        # Most of the window must actually contain motion (not dead air)
        if sum(self._active) / self.window < self.active_ratio:
            return False
        # Count direction changes between consecutive non-zero flows
        flips = 0
        values = list(self._flow_u)
        for left, right in zip(values, values[1:]):
            if left != 0.0 and right != 0.0 and (left * right < 0):
                flips += 1
        return flips >= self.flips_required