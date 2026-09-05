"""Webcam capture wrapper (Phase 2).

Wraps an OpenCV `VideoCapture` in a small class so the webcam loop in
`main.py` (and any future inference code) stays clean and reusable.
"""

import cv2


class CameraFeed:
    """Minimal wrapper around an OpenCV webcam capture."""

    def __init__(self, device_index=0, width=640, height=480):
        # Open the webcam, where 0 = the first webcam found on the system
        self.cap = cv2.VideoCapture(device_index)
        # Many webcams default to 1280x720 or higher, which multiplies the
        # cost of every processing step. Requesting 640x480 here makes the
        # whole pipeline ~4x cheaper; ignored silently when unsupported.
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._is_open = self.cap.isOpened()

    @property
    def is_open(self):
        """True if the webcam opened successfully."""
        return self._is_open

    def read_frame(self):
        """Grab the next frame.

        Returns a tuple ``(ret, frame)`` where ``ret`` is True when a frame
        was captured successfully and ``frame`` is the image itself.
        """
        return self.cap.read()

    def release(self):
        """Release the webcam so other applications can use it.

        Safe to call more than once.
        """
        self.cap.release()
        self._is_open = False