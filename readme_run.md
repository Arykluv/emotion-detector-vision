# How to run this on someone else's computer

A plain-language guide to get the live webcam emotion detector running on a
machine that has never seen this project. Total time: ~10 minutes.

## The fast path (5 steps)

1. **Get the code**

   ```bash
   git clone https://github.com/Arykluv/emotion-detector-vision
   cd emotion-detector-vision
   ```

   No git? Download "Code → Download ZIP" and extract it somewhere simple
   like `C:\emotion-detector`.

   The trained checkpoints are already in the repo (`models/emotion_cnn.pt`
   and `models/emotion_cnn_calibrated.pt`), so **no training is needed**.

2. **Install Python 3.12**

   - Windows: download from https://www.python.org/downloads/ and **tick
     "Add python.exe to PATH"** during install.
   - macOS / Linux: `python3.12` from your package manager, or
     https://www.python.org/downloads/

   The pinned `torch==2.13.0` build targets Python 3.12, so use that version.

3. **Install the dependencies**

   ```bash
   python -m venv .venv
   # activate it:
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   ```

4. **Run it**

   ```bash
   python main.py
   ```

   A window opens showing your mirrored webcam feed. Green boxes follow every
   detected face and label the predicted emotion with its confidence. If the
   model file is missing you will instead see "(no emotion model)" and only
   the rectangles.

## Controls

| Key  | Action                                                  |
| ---- | ------------------------------------------------------- |
| `q`  | quit                                                    |
| `s`  | save a snapshot (PNG)                                   |
| `r`  | start / stop recording (AVI)                            |
| `c`  | dump the session mood log to CSV                        |
| wave | wave at the camera ~1.5 s to close the app              |

Snapshots, recordings and mood CSVs are written to `captures/session-.../`.

## Common options

```bash
python main.py --camera 1              # use the second webcam
python main.py --min-neighbors 8       # stricter face detector
python main.py --model models/emotion_cnn_calibrated.pt   # pick a checkpoint
python main.py --no-tta                # a bit faster, slightly less accurate
python main.py --no-wave-close         # turn off the wave gesture
```

## Troubleshooting

| Symptom                              | Fix                                                     |
| ------------------------------------ | ------------------------------------------------------- |
| `Error: could not open camera 0`     | Camera in use (Teams/Zoom) or blocked; close other apps or try `--camera 1`. On Windows check Settings → Privacy & security → Camera. |
| `Note: no trained model at ...`      | The `models/` folder is missing or incomplete — clone again, or re-download the repo. |
| `Module 'cv2' has no attribute 'CascadeClassifier'` | OpenCV 5 is installed (Haar was removed). `pip install "opencv-python==4.11.0.86"` to force the pinned version. |
| Black or garbled feed                | Camera resolution glitch; try `--camera 1` or restart.  |
| `pip install` errors                 | You are on the wrong Python — run the commands with the 3.12 interpreter, or make sure the venv is activated. |
| Slow or low FPS                      | Expected on a CPU-only laptop. Run `--no-tta`; the CNN expects a webcam-strength machine, not a phone. |

## Make the demo better for *this* person's face (optional)

The model was trained on FER-2013 photos, not webcams. If labels feel wrong
for the new user, re-calibrate on their own face (~5 min):

```bash
python prepare_dataset.py             # once: downloads FER-2013 (needed
                                      # by "finetune", which blends it in)
python calibrate.py collect           # press the emotion's number key, q to save
python calibrate.py finetune
python main.py --model models/emotion_cnn_calibrated.pt
```

Aim for ~100+ samples per emotion so the fine-tune has enough data. Note that
`finetune` reuses the prepared FER-2013 arrays, so run `prepare_dataset.py`
first (it downloads ~30 MB once).

## The no-Python route (for non-technical friends)

You (with Python + `requirements-dev.txt`) can build a standalone Windows app
once and just send the folder:

```bash
pip install -r requirements-dev.txt
build.bat
```

Hand over the whole `dist/emotion-detector/` folder
(including `models/` inside it) to the other person — they only double-click
`emotion-detector.exe`. No Python install needed on their side.

## Advanced / development

These are only needed if you want to change or re-train things:

```bash
# Re-train the 7-class model from scratch (downloads FER-2013 first)
python prepare_dataset.py
python train_model.py

# Run the test suite
pip install -r requirements-dev.txt
python -m pytest tests -q
```