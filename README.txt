================================================================================
  DEEPFAKE DETECTION & FEEDBACK SYSTEM
================================================================================

A tool for analysing videos with ML models (audio + image) to classify
"real vs fake", with an interactive feedback loop for building labelled datasets.
Supports face detection/cropping, sliding-window audio inference, and detailed
result logging.


--------------------------------------------------------------------------------
  PROJECT STRUCTURE
--------------------------------------------------------------------------------

  Deepfake_Detection-Feedbaack_System/
  |
  |-- Program_Workflow/               <- Main runtime code (run everything from here)
  |   |-- main.py                     <- Entry point: detection + feedback pipeline
  |   |-- video_io.py                 <- Audio/frame extraction from video
  |   |-- Facedetection.py            <- MediaPipe face detection & 224x224 crop
  |   |-- models_loader.py            <- Load models + run audio/image predictions
  |   |-- feedback.py                 <- Interactive labelling & dataset builder
  |   |-- retrain.py                  <- Retrain models on accumulated feedback data
  |   |-- best_model.pt               <- Trained image model checkpoint
  |   |-- best_audio_model.pt         <- Trained audio model checkpoint
  |   |-- blaze_face_short_range.tflite  <- MediaPipe face detection model (auto-downloaded)
  |   |-- class_to_idx_img.json       <- Image class label map
  |   |-- class_to_idx_audio.json     <- Audio class label map
  |   |-- dataset_creation/           <- Feedback-labelled dataset (auto-created)
  |   |   |-- audio/  real/  fake/
  |   |   |-- frames/ real/  fake/
  |   |   `-- feedback_log.csv
  |   `-- test-1.mp4                  <- Sample test video
  |
  |-- Audio_Model_Training/           <- Audio model training scripts
  |   |-- Train.py                    <- Main audio training script
  |   |-- dataset_creation.py         <- Audio dataset builder
  |   |-- Run_model.py                <- Quick inference test
  |   `-- results/                    <- Per-run training logs & checkpoints
  |
  |-- Image_Model_Training/           <- Image model training scripts
  |   |-- Train_with_logs.py          <- Main image training script
  |   |-- Testing.py                  <- Evaluation script
  |   `-- logs_img/                   <- Training logs & checkpoints
  |
  `-- README.txt                      <- This file


--------------------------------------------------------------------------------
  FEATURES
--------------------------------------------------------------------------------

  - Video Processing   : Extracts audio (MP3) and frames at configurable FPS
                         using MoviePy with ffmpeg fallback.
  - Face Detection     : Detects the largest face per frame using MediaPipe
                         Tasks API, crops to 224x224 for model input.
  - Audio Inference    : SmallAudioCNN with sliding-window mel-spectrogram
                         inference. Handles clips of any length.
  - Image Inference    : ResNet18 / EfficientNet-B0 on face-cropped frames.
                         Reports top-1 mean confidence across all frames.
  - Result Reports     : Writes aggregate stats (image_result.txt),
                         per-frame predictions (image_aggregate_result.txt),
                         and audio results (audio_result.txt).
  - Feedback Loop      : Shows model predictions, prompts you to label
                         audio/frames as real/fake, and saves them into an
                         organised dataset folder with CSV logging.
  - Retrain            : retrain.py auto-splits the feedback dataset into
                         train/val/test and re-runs both training scripts.
  - GPU Support        : Automatically uses CUDA if available (RTX tested).


--------------------------------------------------------------------------------
  INSTALLATION
--------------------------------------------------------------------------------

  Requirements: Python 3.12, a virtual environment (.venv), ffmpeg on PATH.

  1. Create and activate a virtual environment:
       python3 -m venv .venv
       source .venv/bin/activate          # Linux/Mac
       .venv\Scripts\activate             # Windows

  2. Install PyTorch with CUDA (RTX GPU):
       pip install torch torchvision torchaudio \
           --index-url https://download.pytorch.org/whl/cu121

     CPU-only machines:
       pip install torch torchvision torchaudio

  3. Install remaining dependencies:
       pip install moviepy pillow tqdm opencv-python mediapipe \
                   imageio-ffmpeg scikit-learn

  4. Install ffmpeg system package (if not already present):
       sudo apt install ffmpeg        # Ubuntu/Debian
       brew install ffmpeg            # macOS

  Note: The MediaPipe face detection model (blaze_face_short_range.tflite)
        is downloaded automatically on first run (~224 KB).


--------------------------------------------------------------------------------
  QUICK START
--------------------------------------------------------------------------------

  Run from inside Program_Workflow/:

    cd Program_Workflow

  Analyse the bundled sample video (test-1.mp4):
    python main.py

  Analyse a different video:
    python main.py path/to/your_video.mp4

  Skip the interactive feedback step:
    python main.py your_video.mp4 --no-feedback

  All options:
    python main.py --help


--------------------------------------------------------------------------------
  USAGE
--------------------------------------------------------------------------------

  1. ANALYSE A VIDEO
  ------------------
    python main.py <video_path> [options]

    Options:
      --out-dir DIR        Output directory          (default: outputs)
      --fps FLOAT          Frames per second to extract (default: 2.0)
      --max-frames INT     Max frames sampled for image model (default: 100)
      --topk INT           Top-k classes in output  (default: 3)
      --seed INT           Random seed; -1 = non-deterministic (default: 42)
      --no-feedback        Skip interactive feedback step
      --dataset-root DIR   Feedback dataset folder  (default: dataset_creation)

    Outputs (written to Program_Workflow/):
      image_result.txt          -> aggregate image prediction summary + timings
      image_aggregate_result.txt -> per-frame predictions
      audio_result.txt          -> audio prediction + probability distribution
      outputs/
        audio/         extracted MP3
        frames/        all extracted frames
        frames_cropped/ face-cropped 224x224 frames used for prediction


  2. INTERACTIVE FEEDBACK (runs automatically after detection)
  ------------------------------------------------------------
    After detection, the pipeline prints predictions and asks:

      Add to dataset? [y/n]
      Label the AUDIO as REAL or FAKE? [r/f]
      Label the FRAMES as REAL or FAKE? [r/f]

    Saved files:
      dataset_creation/audio/real/   a1.mp3, a2.mp3, ...
      dataset_creation/audio/fake/   a1.mp3, ...
      dataset_creation/frames/real/  f1.jpg, f2.jpg, ...
      dataset_creation/frames/fake/  f1.jpg, ...
      dataset_creation/feedback_log.csv   (timestamp + labels per run)


  3. RETRAIN MODELS ON FEEDBACK DATA
  ------------------------------------
    Once you have accumulated enough labelled samples:

      python retrain.py                  # retrain both audio + image models
      python retrain.py --mode audio     # audio only
      python retrain.py --mode image     # image only

    Options:
      --dataset-root DIR    Feedback dataset root  (default: dataset_creation)
      --train-split FLOAT   Training fraction      (default: 0.70)
      --val-split FLOAT     Validation fraction    (default: 0.15)
      --epochs INT          Training epochs        (default: 12)
      --batch-size INT      Batch size             (default: 16)
      --seed INT            Random seed            (default: 42)

    The retrained models are saved as:
      best_model.pt          (image model, replaces the existing one)
      best_audio_model.pt    (audio model, replaces the existing one)

    Re-run main.py afterwards to use the updated models.


  4. TRAIN FROM SCRATCH
  ---------------------
    Audio model:
      cd Audio_Model_Training
      python Train.py --data-root <dataset_with_train_val_test> \
                      --epochs 20 --batch-size 16

    Image model:
      cd Image_Model_Training
      python Train_with_logs.py --data-root <dataset_with_train_val_test> \
                                --backbone efficientnet_b0 --epochs 20

    Both scripts expect this folder layout:
      data_root/
        train/  real/  fake/
        val/    real/  fake/
        test/   real/  fake/


--------------------------------------------------------------------------------
  OUTPUT FILES EXPLAINED
--------------------------------------------------------------------------------

  image_result.txt
    - Source video and cropped frames directory
    - Mean top-1 confidence across all face-cropped frames
    - Total and average prediction time

  image_aggregate_result.txt
    - Per-frame breakdown: file path, predicted label, confidence, top-k

  audio_result.txt
    - Predicted label and confidence for the full audio track
    - Full probability distribution over classes
    - Prediction timing

  feedback_log.csv  (in dataset_creation/)
    - One row per feedback session
    - Columns: timestamp, orig_video, audio_label, frames_label,
               num_audio, num_frames, audio_saved_in, frames_saved_in


--------------------------------------------------------------------------------
  MODELS
--------------------------------------------------------------------------------

  Audio model   : SmallAudioCNN (custom CNN)
    Input       : Log-mel spectrogram (1 x 64 mels x T), normalised per sample
    Inference   : Sliding window (4s segments, 2s hop) averaged over windows
    Checkpoint  : best_audio_model.pt

  Image model   : ResNet18 or EfficientNet-B0 (configurable at training time)
    Input       : 224x224 face crop, ImageNet normalisation
    Inference   : Per-frame top-1 confidence, mean aggregated across frames
    Checkpoint  : best_model.pt

  Face detector : MediaPipe Blaze Face (Short Range)
    Model file  : blaze_face_short_range.tflite (auto-downloaded on first run)
    Usage       : Detects largest face per frame, crops with 12% margin


--------------------------------------------------------------------------------
  TROUBLESHOOTING
--------------------------------------------------------------------------------

  No faces detected
    -> Video may have low resolution or unusual angles.
       Lower min_conf in FaceDetector (default 0.5) in Facedetection.py.

  Audio extraction fails
    -> Ensure ffmpeg is installed: ffmpeg -version
       The pipeline falls back to ffmpeg automatically if MoviePy fails.

  CUDA not available
    -> Check nvidia-smi and that the cu121 PyTorch build is installed.
       Falls back to CPU automatically.

  Not enough samples to retrain
    -> retrain.py requires at least 4 samples per modality (one per split).
       Run main.py on more videos and label them first.

  MediaPipe model not found
    -> Delete blaze_face_short_range.tflite and re-run; it will re-download.


--------------------------------------------------------------------------------
  DEPENDENCIES
--------------------------------------------------------------------------------

  torch            2.5.1+cu121    Deep learning framework
  torchvision      0.20.1+cu121   Image model backbones
  torchaudio       2.5.1+cu121    Audio loading & transforms
  opencv-python    4.13.0         Image reading & face cropping
  mediapipe        0.10.33        Face detection (Tasks API)
  moviepy          2.2.1          Video processing & frame extraction
  pillow           11.3.0         Image I/O
  tqdm             4.67.3         Progress bars
  imageio-ffmpeg   0.6.0          Bundled ffmpeg binary
  scikit-learn     1.8.0          ROC-AUC / AUPRC metrics (training)
  numpy            2.4.3          Array operations


================================================================================
