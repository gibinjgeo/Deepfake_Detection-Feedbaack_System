# Deepfake Detection & Feedback System

A multimodal deepfake detection pipeline that analyzes both **audio** and **video frames** of a video, with a human-in-the-loop feedback system for continuous model improvement.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?logo=streamlit)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.33-0097A7?logo=google)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This system detects deepfakes by analyzing a video from two independent modalities:

- **Audio** — A custom CNN (SmallAudioCNN) classifies log-mel spectrograms extracted from the audio track using a sliding-window approach.
- **Image** — A ResNet18 or EfficientNet-B0 backbone classifies 224×224 face crops detected per frame via MediaPipe BlazeFace.

After analysis, users can label the video (real/fake) through either a **Streamlit web GUI** or a **CLI prompt**. Labeled samples are saved to a structured feedback dataset. When enough samples accumulate, the models are automatically fine-tuned on the new data — no original training data required.

---

## System Architecture

```
Video Input (.mp4 / .mov / .avi)
         │
         ▼
┌─────────────────────┐
│   video_io.py       │  Split → Audio (MP3) + Frames (JPG @ N fps)
└──────────┬──────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌────────────────┐
│  Audio  │  │ Facedetection  │  MediaPipe BlazeFace → 224×224 crops
│ (MP3)   │  │    .py         │
└────┬────┘  └──────┬─────────┘
     │               │
     ▼               ▼
┌──────────────┐  ┌────────────────┐
│SmallAudioCNN │  │ ResNet18 /     │
│ Sliding-     │  │ EfficientNet-B0│
│ window infer │  │ Per-frame infer│
└──────┬───────┘  └───────┬────────┘
       │                  │
       └────────┬──────────┘
                ▼
       ┌─────────────────┐
       │   Results +     │  audio_result.txt
       │   Predictions   │  image_result.txt
       └────────┬────────┘  image_aggregate_result.txt
                │
                ▼
       ┌─────────────────┐
       │  feedback.py /  │  User labels: REAL / FAKE
       │    app.py GUI   │  (audio + frames independently)
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ dataset_creation│  audio/{real,fake}/  frames/{real,fake}/
       │   (labeled)     │  feedback_log.csv
       └────────┬────────┘
                │  threshold met?
                ▼
       ┌─────────────────┐
       │ retrain_module  │  Fine-tune on feedback data
       │  .py            │  (warm start, frozen early layers)
       └─────────────────┘
```

---

## Features

- **Dual-modality detection** — independent audio and image analysis on every video
- **Sliding-window audio inference** — handles any length audio without truncation
- **BlazeFace face detection** — lightweight MediaPipe model; skips frames with no faces
- **Streamlit web GUI** — upload videos, view results, label data, trigger fine-tuning from a browser
- **Human-in-the-loop feedback** — label audio and frames independently; labels saved with full audit trail
- **Automatic fine-tuning** — triggers when feedback dataset reaches threshold; backs up old model automatically
- **Frozen-backbone fine-tuning** — prevents overfitting on small feedback datasets
- **Comprehensive logging** — per-epoch CSV/JSON logs, batch timing, GPU memory stats
- **GPU accelerated** — AMP (mixed precision), cuDNN benchmark, TF32 matmul, persistent DataLoader workers

---

## Project Structure

```
Deepfake_Detection-Feedbaack_System/
│
├── Program_Workflow/               # Runtime inference & GUI
│   ├── app.py                      # Streamlit web GUI  ← START HERE
│   ├── main.py                     # CLI entry point
│   ├── models_loader.py            # Model definitions, loading, inference
│   ├── video_io.py                 # Video → audio + frames extraction
│   ├── Facedetection.py            # MediaPipe face detection + cropping
│   ├── feedback.py                 # Interactive CLI labeling & dataset saving
│   ├── retrain_module.py           # Threshold monitor + automatic fine-tuning
│   ├── retrain.py                  # Full retraining CLI from feedback data
│   ├── best_audio_model.pt         # Trained audio model checkpoint  [not in repo]
│   ├── best_model.pt               # Trained image model checkpoint  [not in repo]
│   ├── class_to_idx_audio.json     # Audio class-to-index mapping
│   ├── class_to_idx_img.json       # Image class-to-index mapping
│   └── dataset_creation/           # Feedback dataset (auto-created)
│       ├── audio/real/  fake/
│       ├── frames/real/ fake/
│       └── feedback_log.csv
│
├── Audio_Model_Training/           # Audio model training pipeline
│   ├── Train.py                    # Full SmallAudioCNN training script
│   ├── Run_model.py                # Standalone CLI audio inference
│   ├── dataset_creation.py         # Unified dataset builder (dedup + split)
│   └── Unified_Audio_Dataset/      # Organized train/val/test splits
│
├── Image_Model_Training/           # Image model training pipeline
│   ├── Train_with_logs.py          # Full ResNet18/EfficientNet training script
│   └── Testing.py                  # GUI batch image predictor
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.12
- NVIDIA GPU with CUDA 12.1 (CPU-only also supported, see note below)
- `ffmpeg` accessible on PATH (required for video processing)

### 1. Clone the repository

```bash
https://github.com/gibinjgeo/Deepfake_Detection-Feedbaack_System.git
cd Deepfake_Detection-Feedbaack_System
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

**GPU (CUDA 12.1):**
```bash
pip install -r requirements.txt
```

**CPU-only:** Edit `requirements.txt` — replace the three torch lines with plain versions and remove `--index-url`:
```
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
```
Then run `pip install -r requirements.txt`.

### 4. Place model checkpoints

Download or copy your trained checkpoints into `Program_Workflow/`:

```
Program_Workflow/
├── best_audio_model.pt
├── best_model.pt
├── class_to_idx_audio.json
└── class_to_idx_img.json
```

The JSON files map class names to indices. Example format:
```json
{"fake": 0, "real": 1}
```

---

## Quick Start

### Streamlit Web GUI (recommended)

```bash
cd Program_Workflow
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

The GUI has three tabs:
| Tab | What it does |
|-----|-------------|
| Analyze Video | Upload a video, run full pipeline, view audio + face predictions |
| Feedback & Labeling | Assign real/fake labels, save to dataset |
| Dataset & Fine-Tune | Track dataset progress, trigger model fine-tuning |

### CLI

```bash
cd Program_Workflow

# Analyze a video (with interactive feedback)
python main.py video.mp4

# Analyze with custom settings
python main.py video.mp4 --fps 3.0 --max-frames 150 --topk 2

# Skip feedback prompt
python main.py video.mp4 --no-feedback

# Check dataset status and trigger fine-tuning manually
python retrain_module.py --check
python retrain_module.py --mode both
```

---

## CLI Reference

### `main.py` — Full pipeline

```
python main.py [video] [options]

Arguments:
  video               Path to input video (default: test-1.mp4)

Options:
  --out-dir DIR       Output directory            (default: outputs)
  --fps FLOAT         Frame extraction rate       (default: 2.0)
  --max-frames INT    Max frames for face detect  (default: 100)
  --topk INT          Top-k classes in output     (default: 3)
  --seed INT          Random seed; -1 = random    (default: 42)
  --no-feedback       Skip interactive labeling
  --dataset-root DIR  Feedback dataset location   (default: dataset_creation)
```

### `retrain_module.py` — Fine-tune monitor

```
python retrain_module.py [options]

Options:
  --mode {audio,image,both}   Which model(s) to fine-tune  (default: both)
  --check                     Print dataset status only, no training
  --force                     Fine-tune even if below threshold
  --min-audio INT             Min audio samples/class      (default: 10)
  --min-image INT             Min image samples/class      (default: 20)
```

### `retrain.py` — Full retraining from feedback data

```
python retrain.py [options]

Options:
  --dataset-root DIR          Feedback dataset location    (default: dataset_creation)
  --mode {audio,image,both}                                (default: both)
  --train-split FLOAT         Train fraction               (default: 0.70)
  --val-split FLOAT           Val fraction                 (default: 0.15)
  --seed INT                                               (default: 42)
  --epochs INT                                             (default: 12)
  --batch-size INT                                         (default: 16)
  --workers INT               DataLoader workers           (default: 4)
```

### `Audio_Model_Training/Run_model.py` — Standalone audio inference

```bash
python Run_model.py audio_file.mp3
python Run_model.py folder/              # batch inference on all audio files
```

---

## Models

### Audio — SmallAudioCNN

A lightweight 4-layer 2D CNN that classifies log-mel spectrograms.

| Component | Detail |
|-----------|--------|
| Input | (B, 1, 64, ~250) — log-mel spectrogram |
| Architecture | 4× Conv2d + BN + ReLU + Pool → Flatten → 2× Linear |
| Parameters | ~2.5M |
| Preprocessing | 16 kHz mono, 4s window, n_fft=1024, hop=256, n_mels=64 |
| Inference | Sliding window (4s window, 2s hop), average logits |
| Training aug | SpecAugment: TimeMasking(24) + FrequencyMasking(8) |

### Image — ResNet18 / EfficientNet-B0

ImageNet-1K pretrained backbone with a replaced classification head.

| Component | Detail |
|-----------|--------|
| Input | (B, 3, 224, 224) — face crop, ImageNet normalized |
| Backbone | ResNet18 (~11.2M params) or EfficientNet-B0 (~5.3M params) |
| Head | Linear(512→2) or Linear(1280→2) |
| Preprocessing | MediaPipe BlazeFace → 224×224 crop (10% margin) |
| Training aug | HorizontalFlip + Rotation(10°) + ColorJitter |

### Training configuration (both models)

| Hyperparameter | Training | Fine-tuning |
|----------------|----------|-------------|
| Optimizer | AdamW | Adam |
| Learning rate | 1e-4 | 1e-5 |
| Weight decay | 1e-4 | — |
| LR schedule | ReduceLROnPlateau(factor=0.5, patience=2) | fixed |
| Loss | CrossEntropyLoss | CrossEntropyLoss |
| AMP | Yes (CUDA) | Yes (CUDA) |
| Epochs | 12–50 | 5 |

---

## Feedback Dataset Format

```
dataset_creation/
├── audio/
│   ├── real/   a1.mp3  a2.mp3  ...
│   └── fake/   a1.mp3  a2.mp3  ...
├── frames/
│   ├── real/   f1.jpg  f2.jpg  ...
│   └── fake/   f1.jpg  f2.jpg  ...
├── feedback_log.csv
└── .counters/
    ├── audio_counter_real.txt
    ├── audio_counter_fake.txt
    ├── frames_counter_real.txt
    └── frames_counter_fake.txt
```

`feedback_log.csv` columns:

| Column | Description |
|--------|-------------|
| `timestamp` | ISO 8601 datetime |
| `audio_label` | `real` or `fake` |
| `frames_label` | `real` or `fake` |
| `num_audio` | Audio files saved this session |
| `num_frames` | Frame images saved this session |
| `audio_saved_in` | Absolute path to audio destination folder |
| `frames_saved_in` | Absolute path to frames destination folder |

---

## Output Files

After running `main.py` or the GUI:

```
outputs/
├── audio/
│   └── <video>.mp3                 Extracted audio (mono, 16 kHz)
├── frames/
│   └── <video>/  frame_*.jpg       Raw extracted frames
├── frames_cropped/
│   └── <video>/  frame_*.jpg       Face-cropped 224×224 images
├── audio_result.txt                Audio prediction + confidence
├── image_result.txt                Image aggregate summary
└── image_aggregate_result.txt      Per-frame predictions
```

---

## Automatic Fine-Tuning

The system automatically fine-tunes models when feedback thresholds are met:

| Modality | Default threshold | Strategy |
|----------|------------------|----------|
| Audio | 10 samples/class | Full model fine-tune, lower LR |
| Image | 20 samples/class | Freeze early backbone layers, fine-tune head |

Fine-tuning:
1. Backs up current checkpoint with timestamp: `model_backups/best_model_YYYYMMDD_HHMMSS.pt`
2. Loads existing weights as warm start
3. Trains for 5 epochs on feedback data (80/20 train/val split)
4. Saves new checkpoint **only if** validation accuracy improved
5. Logs result to `auto_retrain_log.json`

---

## Training Your Own Models

### Audio model

```bash
cd Audio_Model_Training

# 1. Build unified dataset from your audio sources
python dataset_creation.py

# 2. Train
python Train.py \
  --data-root Unified_Audio_Dataset \
  --epochs 30 \
  --batch-size 32 \
  --model cnn_small

# Output: results/<timestamp>/best_audio_model.pt
# Copy to:
cp results/<timestamp>/best_audio_model.pt ../Program_Workflow/best_audio_model.pt
```

### Image model

```bash
cd Image_Model_Training

# Expects: dataset/train/real/, dataset/train/fake/, dataset/val/..., dataset/test/...
python Train_with_logs.py \
  --data-root dataset \
  --model resnet18 \
  --epochs 30 \
  --batch-size 32

# Output: best_model.pt + logs_img/
# Copy to:
cp best_model.pt ../Program_Workflow/best_model.pt
```

---

## Troubleshooting

**No faces detected in any frame**
- The video may not contain close-up face shots (e.g. wide-angle, occlusion, blurry faces).
- Lower the confidence threshold in `Facedetection.py`: `FaceDetector(min_conf=0.3)`.
- The audio model still runs independently — image detection is optional.

**`No audio stream found or extraction failed`**
- Confirm `ffmpeg` is installed: `ffmpeg -version`.
- The system attempts MoviePy first, then a direct `ffmpeg` subprocess fallback.
- Some video formats have no audio track — this is handled gracefully.

**`best_audio_model.pt` / `best_model.pt` not found**
- Place trained checkpoint files inside `Program_Workflow/`.
- The GUI will show a warning in the sidebar if they are missing.

**CUDA out of memory**
- Reduce `--batch-size` in training scripts.
- Inference uses a single sample at a time and is not memory-intensive.

**MediaPipe / absl log spam on startup**
- The `WARNING: All log messages before absl::InitializeLog()...` lines are normal. They come from MediaPipe's C++ backend and do not indicate errors.

**`streamlit: command not found`**
- Run `pip install streamlit` or ensure your virtual environment is activated.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep learning | PyTorch 2.5, torchvision, torchaudio |
| Face detection | MediaPipe 0.10 (BlazeFace Tasks API) |
| Video processing | MoviePy 2.x + ffmpeg |
| Image processing | OpenCV 4.x, Pillow |
| Audio processing | torchaudio (MelSpectrogram, SpecAugment) |
| Web GUI | Streamlit 1.56 |
| Data analysis | pandas, NumPy, scikit-learn |
| Training utilities | tqdm, AMP (GradScaler), cuDNN benchmark |

---

## License

This project is licensed under the MIT License.
