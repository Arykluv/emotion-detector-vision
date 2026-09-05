"""Live emotion classification (Phase 6).

Loads the CNN trained by ``train_model.py`` and exposes a tiny interface to
label cropped faces as happy / sad / angry from the webcam loop.

The preprocessing replicates ``prepare_dataset.py``: 48x48 grayscale images
normalized to the range [0, 1]. Callers should pass the same grayscale image
used for face detection, so per-face color conversions are avoided and many
faces can be classified in a single batched model call.
"""

import cv2
import torch
import torch.nn.functional as F

from src.emotion_model import build_model

INPUT_SIZE = 48  # must match the 48x48 images used during training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmotionClassifier:
    """Classify cropped face images with a trained emotion model."""

    def __init__(self, model_path, tta=True):
        # Loading the checkpoint runs on CPU even when inferring on the GPU.
        # Old checkpoints (no "architecture" key) are treated as the "cnn".
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.class_names = list(checkpoint["class_names"])
        architecture = checkpoint.get("architecture", "cnn")
        self.model = build_model(
            num_classes=len(self.class_names),
            architecture=architecture,
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(DEVICE)
        self.model.eval()
        # Test-time augmentation: average each prediction with its mirrored
        # version -> a bit more robust, at double the forward passes
        self.tta = tta

    def predict_batch(self, faces_gray, detailed=False):
        """Classify a list of gray face crops in one batched forward pass.

        Returns a list of ``(label, confidence)`` tuples, or -- when
        ``detailed=True`` -- ``(label, confidence, second_label, second_conf)``
        so callers can tell when the top two classes are nearly tied.
        """
        if not faces_gray:
            return []
        tensors = []
        for face in faces_gray:
            # Match the training preprocessing: 48x48 grayscale in [0, 1]
            resized = cv2.resize(
                face, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA
            )
            tensors.append(torch.tensor(resized, dtype=torch.float32, device=DEVICE))
        batch = torch.stack(tensors).unsqueeze(1) / 255.0  # (N, 1, 48, 48)

        with torch.no_grad():
            prob = F.softmax(self.model(batch), dim=1)
            if self.tta:
                flipped = torch.flip(batch, dims=[3])
                prob = (prob + F.softmax(self.model(flipped), dim=1)) / 2
        results = []
        for row in prob:
            order = row.argsort(descending=True)
            index = int(order[0].item())
            label = self.class_names[index]
            confidence = float(row[index].item())
            if detailed:
                second = int(order[1].item())
                results.append((label, confidence,
                                self.class_names[second], float(row[second].item())))
            else:
                results.append((label, confidence))
        return results

    def predict(self, face_gray):
        """Return (label, confidence) for one cropped grayscale face image."""
        label, confidence = self.predict_batch([face_gray])[0]
        return label, confidence