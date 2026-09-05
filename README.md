# Emotion Detector

A **computer-vision project** that detects faces in a live webcam feed with
OpenCV and classifies the expressed emotion (angry / disgust / fear / happy /
sad / surprise / neutral) with a CNN trained in PyTorch on the FER-2013 dataset.

> Status: the demo runs in real time against all **seven** FER-2013 emotions
> with multi-face tracking, a rolling session mood log and recording hotkeys.
> Baseline: ImageNet-pretrained ResNet-18 **66.02% test accuracy** on the
> seven classes, and a model fine-tuned on your own webcam face is ready at
> `models/emotion_cnn_calibrated.pt` (see [Live demo](#1-live-webcam-demo)
> and [Self-calibration](#5-self-calibration-on-your-own-face)).

## Pipeline

```
Webcam ──► OpenCV ──► Haar Cascade face detection ──► CNN emotion classifier ──► overlay
                │              │                              │
                │        green rectangle per face      all 7 emotions + confidence
                └──── mirrored live feed drawn on screen (multi-face tracking)
```

* **Webcam capture** — `src/camera.py` wraps OpenCV's `VideoCapture`.
* **Face detection** — `src/face_detector.py` wraps OpenCV's Haar Cascade
  (classical CV, zero deep learning).
* **Multi-face tracking** — `src/tracking.py` matches detections across frames
  with IoU so labels follow the **person** (stable ids, survive occlusion,
  never swap between two nearby faces).
* **Emotion classification** — either the small CNN (`src/emotion_model.py`
  `EmotionCNN`) or a pretrained ResNet-18 (`build_model`), trained in
  `train_model.py`, evaluated on the FER-2013 test split.

## Features

* Live, mirrored webcam feed with a green rectangle around every detected face
* **Live emotion label per face** — one of the seven emotions with a
  confidence score and per-person stability (labels only appear after the
  model has agreed across several classification cycles)
* **Multi-face tracking** — stable per-person ids; labels never swap between
  two nearby faces and survive brief occlusions
* **Session mood log** — a rolling bar chart of the emotions seen in the last
  few seconds (`c` dumps it to CSV)
* **Live FPS meter** and a face counter; locality hints kept crisp via a
  half-resolution detection + batching every 3rd frame
* **Hotkeys:** `q` quit · `s` save a snapshot · `r` record/stop an AVI ·
  `c` dump the mood log
* **Wave to close** — wave at the camera (≈1.5 s of reversing horizontal
  motion) and the app closes itself; `--no-wave-close` disables it
* Clean release of the camera, windows and any open recording on exit
* Selectable camera index, detector strictness, model path and TTA from the
  command line
* Graceful fallback to face-detection-only when no trained model exists
* Reproducible dataset preparation (FER-2013 → all **seven** emotions → 48×48
  grayscale, normalized to `[0, 1]`)
* Seedable, self-contained PyTorch training script with per-class metrics,
  a confusion matrix and a `--architecture` switch (ResNet-18 / small CNN)
* **Self-calibration** (`calibrate.py`) — fine-tune on your own webcam face
* A real pytest suite (`tests/`) and PyInstaller packaging (`build.bat`)

## Project structure

```
emotion-detector/
├── main.py              # Entry point: live webcam demo (tracking + mood log)
├── prepare_dataset.py   # Downloads + preprocesses FER-2013 (all 7 emotions)
├── train_model.py       # Trains and evaluates the emotion model
├── calibrate.py         # Self-calibration: collect + fine-tune on your face
├── build.bat            # PyInstaller packaging into dist/
├── requirements.txt     # Pinned runtime dependencies
├── requirements-dev.txt # Pinned dev dependencies (pytest, pyinstaller)
├── README.md            # This file
├── src/
│   ├── camera.py        # CameraFeed - webcam capture wrapper
│   ├── face_detector.py # FaceDetector - Haar Cascade face detection
│   ├── tracking.py      # FaceTracker - IoU multi-face tracking
│   ├── gesture.py       # WaveDetector - wave-to-close optical flow gesture
│   ├── emotion_classifier.py # EmotionClassifier - loads the trained
│   │                          #   model and labels a face crop in real time
│   └── emotion_model.py # build_model - small CNN or ResNet-18 classifier
├── models/              # Trained weights + evaluation artifacts (gitignored)
├── dataset/             # FER-2013 CSV + prepared train/test data (gitignored)
├── captures/            # Snapshots, recordings and mood CSVs (gitignored)
└── tests/               # pytest suite (dataset, train, tracking, main, classifier)
```

## Installation

Requires **Python 3.12** (the `torch` build and the OpenCV 4.x pin are tested
against it). The CNN is tiny, so the **CPU** build of PyTorch is sufficient.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2) Install the pinned dependencies
pip install -r requirements.txt
```

`requirements.txt` pins `opencv-python==4.11.0.86`, `torch==2.13.0`,
`torchvision==0.28.0` and `numpy==2.5.2`. `torchvision` provides the
pretrained ResNet-18 backbone used for the accuracy upgrade.

## Usage

### 1. Live webcam demo

```bash
python main.py
```

A window opens with your mirrored feed. Green boxes track detected faces and
each box is labelled with the predicted emotion (one of the seven) and its
confidence. The top-left area shows a live **FPS meter**, a face counter and a
**rolling mood bar** of the emotions seen in the last few seconds.

Keys:

```text
q  quit              r  start / stop recording (AVI)
s  save a snapshot   c  dump the session mood log to CSV
```

Snapshots, recordings and mood logs land in `captures/<session>/`. Wave at
the camera for about a second and a half and the app will **close itself**
(an optical-flow detector in `src/gesture.py` counts how often the dominant
horizontal motion changes direction — no extra dependencies). Press `q`
to exit.

The app refuses to guess: labels only appear after the model has agreed across
several classification cycles, uncertain predictions are shown compound (e.g.
`sad/angry`), and very low confidence produces no label at all. This keeps the
notorious angry-vs-sad confusion from printing confident wrong answers. Labels
also follow the **person**, not the box: `src/tracking.py` matches faces with
IoU so two faces side by side keep their own emotions.

If `models/emotion_cnn.pt` is missing (no model trained yet), the demo still
runs but only draws the face rectangles and a "(no emotion model)" note.

Extra options:

```bash
python main.py --camera 1                      # use the second webcam
python main.py --min-neighbors 8               # stricter detector (fewer false positives)
python main.py --model path/to/model.pt        # use another trained checkpoint
python main.py --no-tta                        # disable mirror-averaging at run time
python main.py --no-wave-close                 # disable the wave-to-close gesture
```

### 2. Prepare the dataset

```bash
python prepare_dataset.py
```

Downloads FER-2013 on first run (the canonical host is gone from public
access, so the script tries several mirrors), keeps **all seven emotions**
(`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`), splits
into `train` (28,709 images) and `test` (7,178 images), and writes 48×48
grayscale PNGs plus normalized NumPy arrays to `dataset/prepared/`.

### 3. Train the CNN

```bash
python train_model.py
```

The script trains the model, stops early when validation accuracy plateaus,
evaluates on the held-out test set, and saves:

* `models/emotion_cnn.pt` — trained weights + class labels + architecture
* `models/emotion_cnn_evaluation.txt` — final test metrics
* `models/emotion_cnn_confusion_matrix.md` — markdown table for the README

The default is **ImageNet-pretrained ResNet-18** with class-weighted loss,
label smoothing, data augmentation and a warmup/cosine learning-rate schedule
— the recipe behind the 66% seven-class baseline. Notable options:

```bash
python train_model.py                 # recommended: resnet18 + pretrained
python train_model.py --architecture cnn       # original small model
python train_model.py --no-pretrained          # ResNet-18 from scratch
python train_model.py --label-smoothing 0      # disable label smoothing
```

Common options: `--epochs`, `--batch-size`, `--lr`, `--seed` (all seeded by
default for reproducibility). The current `models/emotion_cnn.pt` was
retrained on the **seven-class** dataset (66.02% test accuracy); the fresh
metrics live in `models/emotion_cnn_evaluation.txt` and
`models/emotion_cnn_confusion_matrix.md`.

### 4. Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The suite covers dataset preparation, training helpers, confidence gating and
mood helpers, the multi-face tracker and the runtime classifier.

### 5. Self-calibration on your own face

FER-2013 is "in the wild" studio data; a webcam face is a different
distribution. The quickest accuracy win for your own camera is to fine-tune on
your own face:

```bash
# a) tag your own expressions (press the emotion's number key, q to save)
python calibrate.py collect

# b) fine-tune the base model on FER-2013 + your faces (runs on CPU fine)
python calibrate.py finetune

# c) run the calibrated model
python main.py --model models/emotion_cnn_calibrated.pt
```

Collect ~100+ samples per emotion so the fine-tune has enough to work with.
The custom set must be collected with the same base model (or the default)
so the class order matches.

> A calibration run has already been done for this machine — a fine-tuned
> checkpoint is available at `models/emotion_cnn_calibrated.pt`, so you can
> skip straight to `python main.py --model models/emotion_cnn_calibrated.pt`.

### 6. Build a standalone executable

```bash
pip install -r requirements-dev.txt
build.bat
```

This packages `main.py` into `dist/emotion-detector.exe` with PyInstaller
(collect your trained model next to it, or pass `--model` at run time).

## Machine-learning pipeline

1. **Source data** — FER-2013: 35,887 in-the-wild 48×48 grayscale images of
   faces labelled with one of seven emotions.
2. **Filtering** — all **seven** FER-2013 emotions are kept, in their native
   id order: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`
   (label ids are already contiguous 0–6, so the mapping is the identity).
3. **Splitting** — the `Training` rows become the training set (28,709);
   `PublicTest` + `PrivateTest` become the test set (7,178).
4. **Preprocessing** — images are reshaped to 48×48 grayscale and normalized
   so every pixel value lies in `[0, 1]`.
5. **Training** — a ResNet-18 (ImageNet-pretrained) is fine-tuned with
   class-weighted, label-smoothed cross-entropy, AdamW, data augmentation and
   a warmup/cosine schedule; 10% of the training set is used for
   validation-based early stopping.
6. **Evaluation** — the saved model is scored on the unseen test set: overall
   accuracy/loss, per-class precision/recall/F1 and a confusion matrix.

Data augmentation (mirror, rotation, shift, brightness/contrast jitter) is
applied on every training batch to help the model generalise.

## Model architecture

`build_model()` in `src/emotion_model.py` selects between two families:

* **`resnet18` (default)** — the standard 18-layer residual network with
  ImageNet-pretrained weights, adapted to 48×48 grayscale input (the input
  convolution's filters are averaged over the RGB channels, and the final
  layer is replaced with an N-way output matching the class count). ~11M
  parameters; the accuracy upgrade behind the baseline test score.
* **`cnn` (original)** — the small hand-built network shown below. ~1.3M
  parameters; kept for comparison and CPU speed.

For reference, the original `cnn` architecture (final Dense layer is N-way,
N = number of classes, 7 after the recent dataset change):

| Layer block                          | Output shape     | Notes                          |
| ------------------------------------ | ---------------- | ------------------------------ |
| Input                                | (1, 48, 48)      | grayscale, normalized to [0,1] |
| Conv2d 32 (3×3) + BatchNorm + ReLU   | (32, 48, 48)     | low-level edges / textures     |
| MaxPool2d (2×2)                      | (32, 24, 24)     | reduces spatial size           |
| Conv2d 64 (3×3) + BatchNorm + ReLU   | (64, 24, 24)     | mid-level features             |
| MaxPool2d (2×2)                      | (64, 12, 12)     |                                |
| Conv2d 128 (3×3) + BatchNorm + ReLU  | (128, 12, 12)    | higher-level features          |
| MaxPool2d (2×2)                      | (128, 6, 6)      |                                |
| Flatten                              | 4,608            |                                |
| Dropout (0.3)                        | 4,608            | regularization against         |
| Dense 256 + ReLU                     | 256              | overfitting during training    |
| Dropout (0.5)                        | 256              |                                |
| Dense N                              | N                | class logits (softmax at       |
|                                      |                  | prediction time)               |

Batch normalization stabilises training; the two dropout layers reduce
overfitting, which is important because FER-2013 is relatively small.

## Model evaluation results

Results on the held-out FER-2013 test set (7,178 images), reproduced with
`python train_model.py --seed 42` on the **seven-class** dataset. The
checkpoint is the **ImageNet-pretrained ResNet-18** with class weighting,
label smoothing, augmentation and a warmup/cosine schedule. Full details are
in `models/emotion_cnn_evaluation.txt`.

| Metric        | Value   |
| ------------- | ------- |
| Test accuracy | 66.02%  |
| Test loss     | 1.2527  |

Per-class metrics (class order follows `dataset/prepared/class_names.npy`):

| Class    | Precision | Recall | F1     |
| -------- | --------- | ------ | ------ |
| angry    | 0.5885    | 0.5866 | 0.5876 |
| disgust  | 0.4969    | 0.7117 | 0.5852 |
| fear     | 0.5213    | 0.5020 | 0.5114 |
| happy    | 0.8498    | 0.8546 | 0.8522 |
| sad      | 0.5389    | 0.5060 | 0.5219 |
| surprise | 0.7978    | 0.7882 | 0.7930 |
| neutral  | 0.6006    | 0.6342 | 0.6170 |

`happy` and `surprise` stay the easiest classes; `disgust` has high recall
(0.71) because it is an all-or-nothing class in the raw data (only ~111 test
images) — its low precision (0.50) shows the model over-predicts it. The
facial-expression confusers `sad` / `angry` / `neutral` sit around 0.5–0.6 F1.

## Confusion matrix

Rows = true label, columns = predicted label (generated by `train_model.py`,
stored in `models/emotion_cnn_confusion_matrix.md`):

| (true \\ pred)   | angry | disgust | fear | happy | sad | surprise | neutral |
| ---------------- | ----- | ------- | ---- | ----- | --- | -------- | ------- |
| angry            | 562   | 21      | 117  | 37    | 107 | 18       | 96      |
| disgust          | 20    | 79      | 2    | 0     | 5   | 2        | 3       |
| fear             | 100   | 11      | 514  | 40    | 176 | 72       | 111     |
| happy            | 42    | 8       | 33   | 1516  | 48  | 39       | 88      |
| sad              | 138   | 22      | 163  | 72    | 631 | 17       | 204     |
| surprise         | 19    | 4       | 82   | 33    | 20  | 655      | 18      |
| neutral          | 74    | 14      | 75   | 86    | 184 | 18       | 782     |

`sad` leaks toward `neutral` (204), `angry` and `fear` (138 / 163) — the
classic FER-2013 confusion pattern; `fear` similarly spills into `sad` (176)
and `neutral` (111). Class weighting correctly avoids crushing tiny `disgust`
(only 111 test samples), but its low precision means many other classes'
samples get pulled in.

## Reproducing the training

Run every step from scratch on a machine with Python 3.12:

```bash
# 1) Virtual environment and pinned dependencies
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2) Build the dataset (downloads FER-2013 on first run)
python prepare_dataset.py

# 3) Train and evaluate (seeded for reproducibility)
python train_model.py --seed 42
```

Point-to-point requirements for identical results:

* `opencv-python==4.11.0.86`, `torch==2.13.0`, `torchvision==0.28.0`,
  `numpy==2.5.2` (pinned)
* default hyper-parameters (35-epoch cap, early-stopping patience 6,
  batch 64, AdamW lr 5e-4 / weight decay 1e-4, label smoothing 0.1,
  class-weighted loss, warmup + cosine schedule, seeded at 42)
* FER-2013 mirror downloaded by `prepare_dataset.py` (the canonical source is
  used for the train/test split; tiny count differences between mirrors are
  expected)

## Limitations

* **Facial-expression classification is inherently hard**: expressions are
  subjective, culturally dependent and often ambiguous. FER-2013 itself is
  noisy, with many mislabeled samples.
* **Single source of data** — the model is only exposed to FER-2013's
  in-the-wild style images (low contrast, varied poses) resized to 48×48.
* **Frontal faces only** — both the Haar cascade and the CNN assume a
  roughly frontal, upright face; profile views, heavy occlusion, strong
  lighting changes or sunglasses degrade performance sharply.
* **Confusable classes** — `sad` and `angry` share many facial patterns;
  webcam lighting further shifts the distribution away from FER-2013 (this is
  exactly why `calibrate.py` exists).
* **Harder problem:** 7-class accuracy (66%) is well below the 81% the easy
  3-class subset reached; `disgust` keeps a high recall precisely because it
  is rare — the model over-predicts it, which drags its precision down to 0.50.
* **Self-calibration helps but is not night-vision** — fine-tuning on your own
  face lifts label quality substantially on your own camera, but accuracy on
  any single frame is still far too low for monitoring, screening or other
  decisions about people.

## Troubleshooting

* **`Module 'cv2' has no attribute 'CascadeClassifier'`** — you have OpenCV 5
  installed, which removed Haar cascades. Install the pinned version:
  `pip install "opencv-python==4.11.0.86"`.
* **"Error: could not open camera 0"** — the camera is in use by another app
  (Zoom, Teams), blocked by a privacy switch, or does not exist. Close other
  apps, or pass `--camera 1` / `--camera 2`.
* **Windows camera privacy** — allow camera access for Python under
  Settings > Privacy & security > Camera.
* **Black or garbled feed** — force a common resolution by adding
  `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` and
  `cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)` in `src/camera.py`.
* **Dataset download fails** — the canonical FER-2013 host is no longer
  publicly accessible; `prepare_dataset.py` falls back to mirrors. You can
  also place a valid `fer2013.csv` (columns `emotion, Usage, pixels`) at
  `dataset/fer2013.csv` and rerun.
* **`torch` version mismatch** — make sure you installed
  `requirements.txt` inside the same Python you use to run the scripts.

## Roadmap

* ~~3-class subset~~ → all seven FER-2013 emotions (done)
* ~~Real-time optimisations~~ (half-res detection, batched inference, TTA)
* ~~Multi-person tracking with stable ids~~ (done)
* ~~Recording hotkeys, FPS meter, session mood log~~ (done)
* ~~Self-calibration fine-tuning on your own face~~ (done)
* ~~Real pytest suite~~ (done)
* ~~PyInstaller packaging~~ (done)
* ~~7-class baseline retrain (66.02% test accuracy)~~ (done)
* ~~Calibration run on this machine~~ (done)
* Optional: validation-tuned confidence thresholds, expression-duration
  analytics from the mood log