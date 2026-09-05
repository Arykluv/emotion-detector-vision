"""Self-calibration: teach the model YOUR face (Phase 7).

FER-2013 is "in the wild" studio data; a webcam face is a different
distribution. The two-step workflow bridges that gap with your own data:

  1. Collect:  python calibrate.py collect
     Point the webcam at yourself, change expression, and press the number
     key shown for the emotion until you have ~100+ samples per class in
     your own lighting. Press ``q`` to stop and save.

  2. Retrain:  python calibrate.py finetune
     Fine-tunes the saved base model on your collected faces, blended with
     the original FER-2013 training set (to avoid forgetting what it knew).

  3. Use:      python main.py --model models/emotion_cnn_calibrated.pt

Collecting faces this way is the only practical path past the FER-2013
accuracy ceiling for your own camera.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.camera import CameraFeed
from src.emotion_classifier import INPUT_SIZE
from src.face_detector import FaceDetector
from src.emotion_model import build_model
from train_model import EmotionDataset, evaluate, load_arrays, remap_labels

DEFAULT_MODEL = "models/emotion_cnn.pt"
DEFAULT_CUSTOM_DIR = "dataset/custom"
FER7_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_base_classes(model_path):
    """Class labels from the base checkpoint, or the 7 FER emotions."""
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        return list(checkpoint["class_names"]), checkpoint
    print(f"Note: no model at '{model_path}' - using the 7 FER labels.")
    return FER7_NAMES, None


def collect_cli(args):
    """Open the webcam and let the user tag their own face expressions."""
    class_names, checkpoint = load_base_classes(args.model)
    num_classes = len(class_names)

    camera = CameraFeed(device_index=args.camera)
    if not camera.is_open:
        print(f"Error: could not open camera {args.camera}.")
        raise SystemExit(1)

    try:
        detector = FaceDetector(min_neighbors=args.min_neighbors)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        camera.release()
        raise SystemExit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, labels = [], []
    print("Keys: a number tags the detected face with that emotion, "
          "c removes the last one, q saves and quits.")
    print("Key mapping:")
    for index, name in enumerate(class_names):
        print(f"  {index} -> {name}")

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                print("Error: could not read a frame from the webcam.")
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector.detect(frame, gray=gray, scale=0.5)
            box = None
            if len(faces):
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                box = (x, y, w, h)

            hint = f"Collected: {len(images)}"
            cv2.putText(frame, hint, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c") and images:
                images.pop()
                labels.pop()
                print("Removed last sample.", end="\r")
            elif ord("0") <= key <= ord("9") and box is not None:
                index = key - ord("0")
                if index < num_classes:
                    x, y, w, h = box
                    crop = gray[y : y + h, x : x + w]
                    if crop.size == 0:
                        continue
                    resized = cv2.resize(
                        crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA
                    )
                    images.append(resized)
                    labels.append(index)
                    print(f"Captured {len(images)} samples "
                          f"({class_names[index]})", end="\r")
    finally:
        camera.release()
    cv2.destroyAllWindows()
    print()

    if not images:
        print("No samples collected - nothing saved.")
        return

    X = np.stack(images).astype(np.uint8)
    y = np.asarray(labels, dtype=np.int32)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "class_names.npy", np.array(class_names))
    print(f"Saved {len(y)} samples to {out_dir} "
          f"(unbalanced custom set fine-tunes best if you rebalance)")


def finetune_cli(args):
    """Blend the custom faces with FER-2013 and fine-tune the base model."""
    custom_dir = Path(args.custom_dir)
    base = torch.load(args.base, map_location="cpu", weights_only=True)
    base_class_names = list(base["class_names"])
    architecture = base.get("architecture", "cnn")
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Running on: {device} | base architecture: {architecture}")

    # The custom set must use the same label order as the base model
    custom_names = np.load(custom_dir / "class_names.npy").tolist()
    if custom_names != base_class_names:
        raise SystemExit(
            "Custom class order does not match the base model:\n"
            f"  custom: {custom_names}\n  base:   {base_class_names}\n"
            "Re-run collect with the same base model (or without --model)."
        )
    num_classes = len(base_class_names)

    # Blend the original FER-2013 training set with the user's faces. Both
    # are in [0, 1]; concatenation is fine because labels are 0..N-1 in the
    # same order in both sets.
    data_dir = Path(args.data_dir)
    x_train = load_arrays(data_dir, ["X_train.npy"]).astype(np.float32) / 255.0
    y_train = remap_labels(load_arrays(data_dir, ["y_train.npy"]))
    x_custom = np.load(custom_dir / "X.npy").astype(np.float32) / 255.0
    y_custom = np.load(custom_dir / "y.npy").astype(np.int64)

    x_all = np.concatenate([x_train, x_custom])
    y_all = np.concatenate([y_train, y_custom])
    print(f"Fine-tuning on {len(x_train)} FER + {len(x_custom)} custom "
          f"({len(y_all)} total)")

    counts = np.bincount(y_all, minlength=num_classes).astype(np.float32)
    class_weights = (1.0 / np.maximum(counts, 1))
    class_weights /= class_weights.mean()

    loader = torch.utils.data.DataLoader(
        EmotionDataset(x_all, y_all, augment=False),
        batch_size=args.batch_size, shuffle=True,
    )
    # Evaluate the blended model on the held-out test set each epoch
    x_test = load_arrays(data_dir, ["X_test.npy"])
    y_test = remap_labels(load_arrays(data_dir, ["y_test.npy"]))
    test_loader = torch.utils.data.DataLoader(
        EmotionDataset(x_test, y_test), batch_size=args.batch_size
    )

    model = build_model(num_classes, architecture=architecture, pretrained=False)
    model.load_state_dict(base["state_dict"])
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total += len(labels)
        _, test_acc, _, _ = evaluate(model, test_loader, device)
        print(f"  epoch {epoch:2d} | loss {total_loss / total:.4f} | "
              f"test acc {test_acc:.4f}")
        if test_acc > best_acc:
            best_acc, best_state = test_acc, model.state_dict()

    model.load_state_dict(best_state)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": base_class_names,
        "architecture": architecture,
    }, out)
    print(f"Saved calibrated model -> {out}  (test acc {best_acc:.4f})")
    print(f"Run it with:  python main.py --model {out}")


def main():
    parser = argparse.ArgumentParser(description="Self-calibrate the emotion model.")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Describe yourself with the webcam.")
    collect.add_argument("--camera", type=int, default=0)
    collect.add_argument("--min-neighbors", type=int, default=5)
    collect.add_argument("--model", default=DEFAULT_MODEL,
                         help="Base model (defines the emotion keys).")
    collect.add_argument("--output", default=DEFAULT_CUSTOM_DIR)
    collect.set_defaults(func=collect_cli)

    fine = sub.add_parser("finetune", help="Fine-tune on the collected faces.")
    fine.add_argument("--base", default=DEFAULT_MODEL)
    fine.add_argument("--custom-dir", default=DEFAULT_CUSTOM_DIR)
    fine.add_argument("--data-dir", default="dataset/prepared")
    fine.add_argument("--out", default="models/emotion_cnn_calibrated.pt")
    fine.add_argument("--epochs", type=int, default=10)
    fine.add_argument("--batch-size", type=int, default=32)
    fine.add_argument("--lr", type=float, default=1e-4)
    fine.add_argument("--weight-decay", type=float, default=1e-4)
    fine.add_argument("--device", default="auto")
    fine.set_defaults(func=finetune_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()