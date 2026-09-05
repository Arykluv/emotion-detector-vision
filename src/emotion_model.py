"""Emotion classification models (Phase 5, upgraded).

Two architectures are available through ``build_model()``:

* ``cnn``      - the original small CNN (fast, ~1.3M params, no dependencies).
* ``resnet18`` - ImageNet-pretrained ResNet-18 (the recommended accuracy
                 upgrade; ~11M params, requires ``torchvision``).

Both take 48x48 grayscale images normalized to [0, 1] and output raw logits
per class (happy, sad, angry). ``resnet18`` normalizes internally with
grayscale-averaged ImageNet statistics, so callers always pass [0, 1] inputs.
"""

import torch
import torch.nn as nn

# Grayscale equivalent of the ImageNet mean/std used by pretrained models
IMAGENET_GRAY_MEAN = 0.449
IMAGENET_GRAY_STD = 0.226

ARCHITECTURES = ("cnn", "resnet18")


class EmotionCNN(nn.Module):
    """Small CNN: 3 convolutional blocks followed by 2 fully connected layers.

    Input : a batch of `(batch, 1, 48, 48)` grayscale images, normalized to
            the range [0, 1] (exactly what `prepare_dataset.py` produces).
    Output: raw logits for each class (happy, sad, angry). Apply a softmax
            (or `torch.argmax`) after this when inferring.

    Level of detail used in this architecture:

    Conv block 1: Conv2d 32 filters (3x3) -> BatchNorm -> ReLU -> MaxPool 2x2
    Conv block 2: Conv2d 64 filters (3x3) -> BatchNorm -> ReLU -> MaxPool 2x2
    Conv block 3: Conv2d 128 filters (3x3) -> BatchNorm -> ReLU -> MaxPool 2x2
    Classifier   : Flatten -> Dropout(0.3) -> Dense 256 -> ReLU -> Dropout(0.5)
                   -> Dense 3 (logits)
    """

    def __init__(self, num_classes=3):
        super().__init__()
        # Feature extractor: shrinks the 48x48 input down to 6x6 (128 channels)
        self.features = nn.Sequential(
            # 48x48 -> 24x24
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 24x24 -> 12x12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 12x12 -> 6x6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Classifier: maps the 128 * 6 * 6 features to class logits
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),  # dropout already applied for training
        )

    def forward(self, images):
        """Return raw class logits for a batch of images."""
        return self.classifier(self.features(images))


class NormalizedModule(nn.Module):
    """Wrap a backbone and map [0, 1] grayscale input to ImageNet statistics.

    This keeps the training and inference pipelines identical for every
    architecture: callers pass images in [0, 1] and the model normalizes them
    internally before the backbone sees them.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        images = (images - IMAGENET_GRAY_MEAN) / IMAGENET_GRAY_STD
        return self.backbone(images)


def build_resnet18(num_classes, pretrained=True):
    """ResNet-18 adapted to 48x48 grayscale input, optionally pretrained."""
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:  # torchvision not installed
        raise RuntimeError(
            "The 'resnet18' architecture needs torchvision. "
            'Install it with:  pip install torchvision==0.28.0'
        ) from exc

    net = None
    if pretrained:
        try:
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("Loaded ImageNet-pretrained ResNet-18 weights.")
        except Exception as exc:  # download failed -> fall back to scratch
            print(f"Warning: could not download ImageNet weights ({exc}); "
                  "training from random initialization.")
    if net is None:
        net = resnet18(weights=None)

    # Replace the 3-channel stem with a single grayscale conv, reusing the
    # pretrained filters by averaging them over the RGB dimension
    old_conv = net.conv1
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    net.conv1 = new_conv

    # Swap the final layer for our three emotions
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return NormalizedModule(net)


def build_model(num_classes, architecture="cnn", pretrained=True):
    """Build a classifier for ``num_classes`` given the architecture name."""
    if architecture == "cnn":
        return EmotionCNN(num_classes)
    if architecture == "resnet18":
        return build_resnet18(num_classes, pretrained=pretrained)
    raise ValueError(
        f"Unknown architecture '{architecture}'. Choose from {ARCHITECTURES}."
    )