# Deepfake_Detection-Feedbaack_System


A tool for analyzing videos with ML models (audio + image) to classify "real vs fake", with interactive feedback for building labeled datasets. Supports face detection/cropping and detailed result logging.

## Features

- **Video Processing**: Extract audio (MP3) and frames at configurable FPS using MoviePy + ffmpeg fallback.
- **Face Detection**: Detects largest face per frame with MediaPipe, crops to 224x224 for model input.
- **ML Predictions**: Runs audio and image models; outputs top-1 confidence, top-k classes, timings.
- **Result Reports**: Aggregate stats (`image_result.txt`), per-frame lists (`image_aggregate_result.txt`), audio results (`audio_result.txt`).
- **Interactive Dataset Building**: Review predictions, label audio/frames as real/fake, auto-organize into `dataset_creation/audio|frames/real|fake/` with counters and CSV logging.
- **Configurable**: FPS, JPEG quality, MP3 bitrate, max frames, random sampling, GPU support.

## Quick Demo

```bash
python main.py  # Processes test-1.mp4 (attached), generates outputs/
```

Generates:
```
outputs/
├── audio/test-1.mp3
├── frames/test-1/  # All frames
├── frames/cropped/ # Face-cropped subset
├── image_result.txt     # Summary
├── image_aggregate_result.txt  # Per-frame
└── audio_result.txt
```

For feedback mode (run after `main.py`):
```bash
# Edits main.py to enable FEEDBACKENABLED=True
python main.py  # Launches interactive review/labeling
```

## Installation

1. Clone the repo:
   ```bash
   git clone <your-repo-url>
   cd project-name
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio  # GPU: add --index-url https://download.pytorch.org/whl/cu121
   pip install moviepy pillow tqdm opencv-python mediapipe imageio-ffmpeg
   ```

3. Place your models:
   - Assumes `models/loader.py` (not attached) loads audio/image models via `load_audio_model()`, `load_image_model()`.
   - Update paths in `main.py` if needed.

## Usage

### 1. Analyze a Video
Edit `main.py` settings (e.g., `VIDEOPATH="your-video.mp4"`, `OUTDIR="outputs"`), then:
```bash
python main.py
```
- Skips frames without faces.
- Samples up to `MAXFRAMES=100` randomly.
- Outputs console summary + text files.

### 2. Build Dataset Interactively
Set `FEEDBACKENABLED=True` and `DATASETROOT="dataset_creation"` in `main.py`:
```bash
python main.py
```
- Shows predictions for audio + sample frames.
- Prompts: "Add to dataset? (y/n)", "Label AUDIO/FRAMES: real/fake".
- Saves to organized folders, logs to `feedbacklog.csv`.

## File Structure

```
project/
├── main.py              # Main pipeline + feedback wiring
├── video_io.py          # Audio/frame extraction
├── Facedetection.py     # MediaPipe face crop to 224x224
├── feedback.py          # Interactive labeling/dataset
├── models/loader.py     # Load/predict (add your models here)
├── test-1.mp4           # Sample video
├── *.json               # Model class indices
└── outputs/             # Results (auto-generated)
```

## Configuration (in `main.py`)

| Parameter       | Default     | Description |
|-----------------|-------------|-------------|
| `IMG_FPS`       | 2.0        | Frames per second to extract. |
| `MAXFRAMES`     | 100        | Max sampled frames. |
| `TOPK`          | 3          | Top-k classes in output. |
| `JPEGQUALITY`   | 90         | Image quality. |
| `AUDIOBITRATE`  | 192k       | MP3 bitrate. |

## Results Example

**Console Output**:
```
IMAGE: Top-1 mean confidence across cropped frames: 0.9234
IMAGE: Total prediction time (cropped frames): 2.1456s
AUDIO: Prediction: fake (prob: 0.8765)
```

**Dataset Output**:
```
dataset_creation/
├── audio/
│   ├── real/a001.mp3, a002.mp3, ...
│   └── fake/a001.mp3, ...
├── frames/
│   ├── real/f001.jpg, f002.jpg, ...
│   └── fake/f001.jpg, ...
└── feedbacklog.csv
```

## Troubleshooting

- **No faces detected**: Lower `minconf` in `Facedetection.py` or check video quality.
- **Audio fails**: Ensure ffmpeg installed; tool falls back automatically.
- **GPU**: Set `device='cuda'` if CUDA available.
- **Models**: Implement `predict_image()`/`predict_audio()` in `models/loader.py` returning `{'label': str, 'prob': float, 'dist': dict, 'topk': list}`.
