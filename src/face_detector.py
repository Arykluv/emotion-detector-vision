"""Face detection module (Phase 3).

Wraps OpenCV's Haar Cascade face detector so the webcam loop in `main.py`
stays short. This is a classical computer-vision detector: no deep learning
is involved.
"""

import cv2


class FaceDetector:
    """Detects frontal faces in images using OpenCV's Haar Cascade."""

    def __init__(self, cascade_path=None, min_neighbors=5, scale_factor=1.2):
        # OpenCV 5 removed Haar cascades; this project requires OpenCV 4.x
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "This project needs OpenCV 4.x (OpenCV 5 removed Haar cascades).\n"
                'Fix it with:  pip install "opencv-python==4.11.0.86"'
            )

        # Use OpenCV's bundled frontal-face cascade unless a path is given
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load the cascade file: {cascade_path}")

        # Detection tuning:
        #   scale_factor  - smaller values search more scales (slower, better recall)
        #   min_neighbors - higher values reject more false positives
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, frame, gray=None, scale=1.0):
        """Detect faces; return ``(x, y, width, height)`` rectangles.

        ``gray`` may be a pre-computed grayscale version of ``frame`` to avoid
        converting twice per frame. ``scale`` (0 < scale <= 1) shrinks the
        image before detection, so Haar runs over up to ``scale**2`` fewer
        pixels (e.g. ``0.5`` = ~4x faster); returned boxes are scaled back to
        the original ``frame`` coordinate space.
        """
        # Haar cascades are optimized for grayscale images
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run the cascade on a downscaled image when a detection scale is set
        if scale < 1.0 and scale > 0:
            working = cv2.resize(
                gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        else:
            working = gray

        faces = self.cascade.detectMultiScale(
            working,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
        )

        if working is not gray:
            inverse = 1.0 / scale
            faces = [
                (int(x * inverse), int(y * inverse),
                 int(w * inverse), int(h * inverse))
                for (x, y, w, h) in faces
            ]
        return faces