"""Train and evaluate the emotion CNN (Phase 5).

Loads the prepared dataset from `dataset/prepared`, trains the small CNN in
`src/emotion_model.py`, reports per-class metrics and a confusion matrix on
the held-out test set, and saves:

  - models/emotion_cnn.pt        trained weights and the class labels
  - models/evaluation.txt        final test metrics
  - models/confusion_matrix.md   markdown table, ready for the README

Usage (after `python prepare_dataset.py`):

    python train_model.py                          # resnet18 + pretrained weights
    python train_model.py --architecture cnn       # original small CNN
    python train_model.py --no-pretrained          # no ImageNet weights
    python train_model.py --no-augment             # disable augmentation
    python train_model.py --no-class-weights       # plain cross-entropy loss
    python train_model.py --label-smoothing 0      # disable label smoothing

Tune hyperparameters with the command-line flags or the values below.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from src.emotion_model import ARCHITECTURES, build_model

# Hyperparameters (overridable from the command line)
DEFAULT_EPOCHS = 35
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 6
DEFAULT_SEED = 42
DEFAULT_LABEL_SMOOTHING = 0.1
VAL_RATIO = 0.1  # 10% of the training set is held out for early stopping


def set_seed(seed):
    """Pin every random source so training is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_arrays(data_dir, patterns, required=True):
    """Load a numpy artifact, raising a friendly error if it is missing."""
    for name in patterns:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return np.load(path)
    if required:
        raise FileNotFoundError(
            f"Could not find any of {patterns} in '{data_dir}'.\n"
            "Run `python prepare_dataset.py` first."
        )
    return None


def make_splits(x, y, val_ratio, seed):
    """Split the training arrays into a training and a validation slice."""
    indices = np.random.RandomState(seed).permutation(len(y))
    val_count = int(len(y) * val_ratio)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def remap_labels(labels):
    """Remap class ids to contiguous indices 0..N-1 in sorted class order."""
    in_order = sorted(set(int(v) for v in labels))
    mapping = {label: index for index, label in enumerate(in_order)}
    return np.array([mapping[int(v)] for v in labels], dtype=np.int64)


class EmotionDataset(torch.utils.data.Dataset):
    """Wraps numpy arrays into something a DataLoader can iterate over."""

    def __init__(self, images, labels, augment=False):
        # (N, 48, 48) -> (N, 1, 48, 48); labels are indices into class_names
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.augment:
            image = self.augment_image(image)
        return image, self.labels[index]

    def augment_image(self, image):
        """Apply light, label-preserving augmentation to one image in [0, 1]."""
        rnd = torch.rand  # torch.manual_seed makes this reproducible

        if rnd(()).item() < 0.5:           # mirror (safe for expressions)
            image = TF.hflip(image)

        if rnd(()).item() < 0.4:           # small rotation
            angle = (rnd(()).item() * 2 - 1) * 12  # +-12 degrees
            image = TF.rotate(image, angle, fill=0.0)

        if rnd(()).item() < 0.4:           # small shift via pad + random crop
            image = TF.pad(image, padding=4, fill=0.0)
            image = TF.crop(image,
                            int(rnd(()).item() * 8),
                            int(rnd(()).item() * 8),
                            48, 48)

        if rnd(()).item() < 0.3:           # brightness jitter (0.7x - 1.3x)
            image = TF.adjust_brightness(image, 0.7 + 0.6 * rnd(()).item())
        if rnd(()).item() < 0.3:           # contrast jitter (0.7x - 1.3x)
            image = TF.adjust_contrast(image, 0.7 + 0.6 * rnd(()).item())
        return image


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one pass over the training set and return the mean loss."""
    model.train()
    total_loss, total = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total += len(labels)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (mean_loss, accuracy, true_labels, predictions) on a loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    all_true, all_pred = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)

        total_loss += criterion(logits, labels).item() * len(labels)
        total_correct += (preds == labels).sum().item()
        total += len(labels)

        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    return total_loss / total, total_correct / total, all_true, all_pred


def confusion_matrix(all_true, all_pred, num_classes):
    """Build a num_classes x num_classes matrix (rows = true, cols = predicted)."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(all_true, all_pred):
        matrix[true, pred] += 1
    return matrix


def per_class_metrics(matrix):
    """Return (precision, recall, f1) arrays computed from the confusion matrix."""
    true_cols = matrix.sum(axis=0)
    row_sums = matrix.sum(axis=1)

    precision = np.where(true_cols > 0, matrix.diagonal() / np.maximum(true_cols, 1), 0.0)
    recall = np.where(row_sums > 0, matrix.diagonal() / np.maximum(row_sums, 1), 0.0)
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / np.maximum(precision + recall, 1e-9),
        0.0,
    )
    return precision, recall, f1


def format_confusion_matrix(matrix, class_names):
    """Render the confusion matrix as a markdown table for the README."""
    header = "| " + " | ".join([" "] + class_names) + " |"
    separator = "|" + "---|" * (len(class_names) + 1)
    lines = [header, separator]
    for name, row in zip(class_names, matrix):
        cells = " | ".join([name] + [str(int(v)) for v in row])
        lines.append(f"| {cells} |")
    return "\n".join(lines)


def write_evaluation(path, accuracy, loss, class_names, precision, recall, f1):
    """Write the final scorecard to evaluation.txt."""
    lines = [
        "Emotion classifier - evaluation on the FER-2013 test set",
        "=" * 55,
        f"Test accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)",
        f"Test loss     : {loss:.4f}",
        "",
        "Per-class metrics:",
        f"{'class':>8} {'precision':>10} {'recall':>8} {'f1':>8}",
    ]
    for name, p, r, f in zip(class_names, precision, recall, f1):
        lines.append(f"{name:>8} {p:>10.4f} {r:>8.4f} {f:>8.4f}")
    with open(path, "w", encoding="utf-8") as dest:
        dest.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train the FER-2013 emotion CNN.")
    parser.add_argument("--data-dir", default="dataset/prepared", help="Output dir of prepare_dataset.py.")
    parser.add_argument("--model-out", default="models/emotion_cnn.pt", help="Where to save the trained model.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Max number of epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Samples per batch.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate (AdamW).")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING,
                        help="Cross-entropy label smoothing (0 disables it).")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="resnet18",
                        help="Model family: 'resnet18' (recommended) or 'cnn' (original).")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained", default=True,
                        help="Train ResNet-18 from scratch instead of ImageNet weights.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early-stopping patience (epochs).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")
    parser.add_argument("--device", default="auto", help="'cpu', 'cuda' or 'auto' (default).")
    parser.add_argument("--no-augment", action="store_false", dest="augment", default=True,
                        help="Disable random horizontal-flip augmentation.")
    parser.add_argument("--no-class-weights", action="store_false", dest="class_weights", default=True,
                        help="Use plain cross-entropy instead of class-weighted loss.")
    args = parser.parse_args()

    set_seed(args.seed)

    # Resolve the compute device ("auto" picks CUDA only if it is available)
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Running on: {device}")

    # Load the prepared arrays
    data_dir = Path(args.data_dir)
    x_train = load_arrays(data_dir, ["X_train.npy"])
    y_train = load_arrays(data_dir, ["y_train.npy"])
    x_test = load_arrays(data_dir, ["X_test.npy"])
    y_test = load_arrays(data_dir, ["y_test.npy"])
    class_names = np.load(data_dir / "class_names.npy").tolist()

    # The stored labels are class indices that must be contiguous (0..N-1),
    # matching the CNN output. Old prepared files may contain the raw
    # FER-2013 ids (e.g. 0, 3, 4 instead of 0, 1, 2), which would break the
    # loss function, so remap every label through the sorted class order.
    y_train = remap_labels(y_train)
    y_test = remap_labels(y_test)

    # Per-class weights BEFORE splitting so the weight of the rare angry
    # class is based on the full training set.
    if args.class_weights:
        counts = np.bincount(y_train, minlength=len(class_names)).astype(np.float32)
        inverse = 1.0 / np.maximum(counts, 1)
        class_weights = inverse / inverse.mean()  # mean = 1.0
        print("Class weights (mean 1.0): " + ", ".join(
            f"{n}={w:.3f}" for n, w in zip(class_names, class_weights)
        ))
    else:
        class_weights = None

    # Hold out a validation slice to decide when to stop training
    x_train, y_train, x_val, y_val = make_splits(x_train, y_train, VAL_RATIO, args.seed)
    print(f"train={len(y_train)}  val={len(y_val)}  test={len(y_test)}"
          + ("  | augment=on" if args.augment else "  | augment=off"))

    # Build loaders (augment only the training split)
    train_loader = torch.utils.data.DataLoader(
        EmotionDataset(x_train, y_train, augment=args.augment),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(EmotionDataset(x_val, y_val), batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(EmotionDataset(x_test, y_test), batch_size=args.batch_size)

    # Model, optimizer and loss
    print(f"Architecture: {args.architecture}")
    model = build_model(
        num_classes=len(class_names),
        architecture=args.architecture,
        pretrained=args.pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Learning-rate schedule: linear warm-up, then cosine decay to zero
    warmup_epochs = min(3, args.epochs // 5)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    # Training loop with early stopping on the validation accuracy
    print("Training...")
    best_val_acc, best_state, epochs_waited = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, val_acc, _, _ = evaluate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch:3d} | train loss {loss:.4f} | val acc {val_acc:.4f} "
              f"| lr {lr_now:.2e}")
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            epochs_waited = 0
        else:
            epochs_waited += 1
            if epochs_waited >= args.patience:
                print(f"  early stopping at epoch {epoch}")
                break

    # Restore the weights that performed best on the validation set
    model.load_state_dict(best_state)

    # Final evaluation on the held-out test set
    test_loss, test_acc, all_true, all_pred = evaluate(model, test_loader, device)
    matrix = confusion_matrix(all_true, all_pred, len(class_names))
    precision, recall, f1 = per_class_metrics(matrix)

    # Save every artifact the README references
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "architecture": args.architecture,
        },
        args.model_out,
    )
    write_evaluation(
        os.path.splitext(args.model_out)[0] + "_evaluation.txt",
        test_acc, test_loss, class_names, precision, recall, f1,
    )
    confusion_md = format_confusion_matrix(matrix, class_names)
    with open(os.path.splitext(args.model_out)[0] + "_confusion_matrix.md", "w", encoding="utf-8") as dest:
        dest.write(confusion_md + "\n")

    # Console summary
    print("\n" + "=" * 55)
    print(f"Test accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Per-class f1  : " + ", ".join(f"{n}={f:.4f}" for n, f in zip(class_names, f1)))
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(confusion_md)
    print("=" * 55)
    print(f"\nSaved model        -> {args.model_out}")
    print("Saved evaluation  -> " + os.path.splitext(args.model_out)[0] + "_evaluation.txt")
    print("Saved conf matrix -> " + os.path.splitext(args.model_out)[0] + "_confusion_matrix.md")
    print("\nTip: paste the confusion matrix markdown into the README.")


if __name__ == "__main__":
    main()