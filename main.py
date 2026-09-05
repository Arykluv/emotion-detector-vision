"""Emotion detector - main entry point (webcam demo).

Streams the webcam feed, draws a green rectangle around every tracked face
and labels each with its predicted emotion (happy / sad / angry ...) plus a
confidence score.

Features:
  * real-time optimisations: 640x480 capture, half-resolution Haar detection,
    batched CNN inference every few frames, test-time augmentation
  * multi-face tracking: every person gets a stable id (labels never swap
    between two nearby faces and survive brief occlusions)
  * confidence gating + temporal smoothing (no confident wrong answers)
  * a live session mood log (rolling emotion proportions) and capturing:

      q  quit          s  save a snapshot      r  toggle recording
      c  dump the mood log to a CSV

Press ``q`` to quit. Face detection uses OpenCV's Haar Cascade; emotion
classification runs the model trained by ``train_model.py``.

Usage:
    python main.py
    python main.py --camera 1          # use the second webcam
    python main.py --min-neighbors 8   # fewer false positives, may miss faces
    python main.py --model models/emotion_cnn.pt   # which trained model to use
"""

import argparse
import datetime
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2  # OpenCV library for computer vision

from src.camera import CameraFeed
from src.emotion_classifier import EmotionClassifier
from src.face_detector import FaceDetector
from src.tracking import FaceTracker

WINDOW_NAME = "Emotion Detector"
EXIT_SUCCESS = 0
EXIT_CAMERA_ERROR = 1

DEFAULT_MODEL = "models/emotion_cnn.pt"


def default_model_path():
    """Default checkpoint path; resolves next to the app when packaged."""
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "models",
                            "emotion_cnn.pt")
    return DEFAULT_MODEL
DETECT_SCALE = 0.5  # detect faces on a half-resolution copy (4x faster Haar)
CLASSIFY_EVERY = 3  # run the CNN only every Nth frame
STREAK_CONFIRM = 2  # label only shown after this many agreeing classify cycles
MOOD_WINDOW = 240   # how many labelled frames the session mood log keeps

# Confidence gating: refuse to print a confident label when the model is
# uncertain or torn between two classes (e.g. angry vs sad).
MIN_CONFIDENCE = 0.35  # below this, show no label at all
CONFIDENCE_GAP = 0.15  # top1 vs top2 closer than this -> "top1/top2?"

# Bar colours for the mood panel (BGR), indexed by rank
MOOD_PALETTE = [
    (0, 255, 0),    # rank 1  -> green
    (255, 0, 0),    # rank 2  -> blue
    (0, 0, 255),    # rank 3  -> red
    (255, 165, 0),  # orange
    (255, 105, 180),# pink
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
]


def timestamp():
    """Compact local timestamp for capture filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_label(top, top_conf, second, second_conf):
    """Return the displayed label after confidence gating (None if unsure)."""
    if top_conf < MIN_CONFIDENCE:
        return None
    if top_conf - second_conf < CONFIDENCE_GAP:
        return f"{top}/{second}"
    return top


def mood_bucket(display):
    """Map a displayed label to its mood-log bucket (ambiguous -> '?')."""
    return "?" if "/" in display else display


def draw_emotion_label(frame, x, y, width, label, confidence):
    """Draw a readable emotion label above a face rectangle."""
    text = f"{label} {confidence:.0%}"
    scale, thickness = 0.6, 2
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    # Keep the label inside the frame (top edge or just above the box)
    label_y = max(y - 8, text_h + 4)
    cv2.rectangle(
        frame,
        (x, label_y - text_h - 4),
        (x + text_w + 6, label_y + baseline),
        (0, 255, 0),
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (x + 3, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
    )


def draw_mood_panel(frame, x, y, counts):
    """Draw the rolling emotion proportions as proportional bars."""
    total = float(sum(counts.values()))
    if total <= 0:
        return
    peak = max(counts.values())
    for rank, (label, count) in enumerate(
        sorted(counts.items(), key=lambda item: -item[1])
    ):
        color = MOOD_PALETTE[rank % len(MOOD_PALETTE)]
        bar_w = max(10, int((count / peak) * 150))
        bar_y = y + rank * 18
        cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + 12), color, -1)
        cv2.putText(
            frame,
            f"{label} {count / total * 100:.0f}%",
            (x + 6, bar_y + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
        )


def parse_args():
    """Read command-line options for the webcam demo."""
    parser = argparse.ArgumentParser(description="Live webcam emotion detection.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (0 = first webcam found).",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Haar detector strictness (higher = fewer false positives).",
    )
    parser.add_argument(
        "--model",
        default=default_model_path(),
        help="Path to a trained model checkpoint (default: models/emotion_cnn.pt).",
    )
    parser.add_argument(
        "--no-tta",
        action="store_false",
        dest="tta",
        default=True,
        help="Disable test-time augmentation (mirror averaging) at run time.",
    )
    return parser.parse_args()


def load_classifier(model_path, tta=True):
    """Load the emotion classifier, or return None and explain when absent."""
    if not Path(model_path).is_file():
        print(f"Note: no trained model at '{model_path}' - running with face "
              f"detection only. Train one first with: python train_model.py")
        return None
    try:
        classifier = EmotionClassifier(model_path, tta=tta)
    except Exception as exc:  # corrupt or incompatible checkpoint
        print(f"Warning: could not load model '{model_path}' ({exc}) - "
              f"running with face detection only.")
        return None
    print(f"Emotion labels: {classifier.class_names}")
    return classifier


def dump_mood_csv(captures_dir, counts):
    """Write the current mood-log counters to a timestamped CSV."""
    path = os.path.join(captures_dir, f"mood_{timestamp()}.csv")
    with open(path, "w", encoding="utf-8") as dest:
        dest.write("label,count\n")
        for label, count in sorted(counts.items(), key=lambda item: -item[1]):
            dest.write(f"{label},{count}\n")
    print(f"Mood log written to: {path}")
    return path


def main():
    args = parse_args()

    # Open the webcam and prepare the face detector
    camera = CameraFeed(device_index=args.camera)
    if not camera.is_open:
        print(f"Error: could not open camera {args.camera}.")
        print("Check that no other app is using it and the device exists.")
        raise SystemExit(EXIT_CAMERA_ERROR)

    try:
        detector = FaceDetector(min_neighbors=args.min_neighbors)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        camera.release()
        raise SystemExit(EXIT_CAMERA_ERROR)

    classifier = load_classifier(args.model, tta=args.tta)
    tracker = FaceTracker()
    captures_dir = os.path.join("captures", timestamp())
    os.makedirs(captures_dir, exist_ok=True)

    mood_history = deque(maxlen=MOOD_WINDOW)
    video_writer = None  # active cv2.VideoWriter, or None while idle
    fps = 0.0
    last_time = time.perf_counter()

    print("Keys: q quit | s snapshot | r record | c mood CSV")
    print("Webcam opened successfully. Press 'q' to quit.")
    try:
        frame_index = 0
        while True:
            # Grab the next frame; ret = True when a frame was captured
            ret, frame = camera.read_frame()
            if not ret:
                print("Error: could not read a frame from the webcam.")
                break

            # Mirror the frame (selfie-style) before drawing on it
            frame = cv2.flip(frame, 1)

            # Grayscale once per frame, reused for detection and classification
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces on a half-resolution copy; boxes are scaled back
            faces = detector.detect(frame, gray=gray, scale=DETECT_SCALE)

            # Run the CNN only every CLASSIFY_EVERY frames, batching all faces
            # into a single call; gate each prediction on confidence.
            labels_for_track = [None] * len(faces)
            if classifier is not None and frame_index % CLASSIFY_EVERY == 0:
                crops = [gray[y : y + h, x : x + w] for (x, y, w, h) in faces]
                results = classifier.predict_batch(crops, detailed=True)
                for index, (top, top_conf, second, second_conf) in enumerate(results):
                    display = stable_label(top, top_conf, second, second_conf)
                    if display is not None:
                        labels_for_track[index] = (display, top_conf)

            # Track faces (stable ids, IoU matching) and feed labels in
            tracks = tracker.update(faces, labels=labels_for_track)
            for track_id, box, label, confidence, streak in tracks:
                x, y, width, height = box
                cv2.rectangle(
                    frame, (x, y), (x + width, y + height), (0, 255, 0), 2
                )
                if label is not None and streak >= STREAK_CONFIRM:
                    draw_emotion_label(frame, x, y, width, label, confidence)
                    mood_history.append(mood_bucket(label))

            # Top-left: face counter
            counter_label = f"Faces: {len(faces)}"
            if classifier is None:
                counter_label += "  (no emotion model)"
            cv2.putText(
                frame, counter_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            # Session mood log (rolling proportions) below the counter
            draw_mood_panel(frame, 10, 55, Counter(mood_history))

            # Top-right: frames per second
            cv2.putText(
                frame, f"FPS: {fps:.0f}", (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            if video_writer is not None:
                cv2.putText(
                    frame, "REC", (frame.shape[1] - 90, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            # Recording: write the processed (labelled) frame to disk
            if video_writer is not None:
                video_writer.write(frame)

            cv2.imshow(WINDOW_NAME, frame)

            # Frame pacing + FPS estimate
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 * (1.0 / dt)
            frame_index += 1

            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                path = os.path.join(captures_dir, f"snapshot_{timestamp()}.png")
                cv2.imwrite(path, frame)
                print(f"Snapshot saved to: {path}")
            elif key == ord("r"):
                if video_writer is None:
                    writer_path = os.path.join(captures_dir, f"rec_{timestamp()}.avi")
                    video_writer = cv2.VideoWriter(
                        writer_path,
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        20.0,
                        (frame.shape[1], frame.shape[0]),
                    )
                    print(f"Recording to: {writer_path}")
                else:
                    video_writer.release()
                    video_writer = None
                    print("Recording stopped.")
            elif key == ord("c"):
                dump_mood_csv(captures_dir, Counter(mood_history))
    finally:
        # Always release the webcam and any open recording, even on errors
        if video_writer is not None:
            video_writer.release()
        camera.release()

    cv2.destroyAllWindows()
    print("Webcam released. Goodbye!")
    raise SystemExit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()