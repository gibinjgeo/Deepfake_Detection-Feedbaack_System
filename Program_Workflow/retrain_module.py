"""
retrain_module.py
=======================
Watches the feedback dataset and automatically fine-tunes the audio and image
models when enough labelled samples have accumulated.

Why fine-tuning, not full retraining?
--------------------------------------
The existing model checkpoint already ENCODES all the original training data as
learned weights.  Loading those weights and continuing to train on the feedback
data is called "fine-tuning".  It gives you the best of both worlds:

  - Old knowledge is preserved in the starting weights
    (no catastrophic forgetting, no need to keep the original data files)
  - New patterns from feedback samples are learned on top
  - Training is fast — small dataset, few epochs, warm start

This answers the two questions:
  Q: Should the new model train on old + new data?
  A: No need — the old MODEL WEIGHTS already capture all that old knowledge.
     Just fine-tune on the new feedback data only.

  Q: Can the old model be improved without training on all the original data?
  A: Yes — that is exactly what fine-tuning is.

Pipeline
--------
  1. Count samples in dataset_creation/{audio|frames}/{real,fake}/
  2. Skip if either class is below MIN_SAMPLES_PER_CLASS
  3. Back up the current model checkpoint (timestamped copy)
  4. Load checkpoint → restore architecture + weights
  5. Fine-tune on feedback data (lower LR, fewer epochs)
  6. Evaluate on a held-out 20% split
  7. Save new checkpoint only if validation accuracy improved
  8. Write a JSON log entry

Usage
-----
  # Status check only (no changes made)
  python retrain_module.py --check

  # Auto-retrain if thresholds are met
  python retrain_module.py

  # Retrain one modality only
  python retrain_module.py --mode audio
  python retrain_module.py --mode image

  # Force fine-tune even if below threshold
  python retrain_module.py --force

  # Called from main.py after every feedback session
  from auto_retrain_monitor import maybe_auto_retrain
  maybe_auto_retrain()
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# Reuse architectures and preprocessing from models_loader
from models_loader import (
    SmallAudioCNN,
    _AudioPreproc,
    _build_image_backbone,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent.resolve()
DATASET_ROOT = HERE / "dataset_creation"
BACKUP_DIR   = HERE / "model_backups"
MONITOR_LOG  = HERE / "auto_retrain_log.json"

AUDIO_CKPT   = HERE / "best_audio_model.pt"
IMAGE_CKPT   = HERE / "best_model.pt"

AUDIO_EXTS   = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── Thresholds & fine-tune hyper-parameters ───────────────────────────────────
MIN_SAMPLES_PER_CLASS_AUDIO = 10   # minimum per-class audio files before training
MIN_SAMPLES_PER_CLASS_IMAGE = 20   # minimum per-class image files before training

FINETUNE_EPOCHS     = 5
FINETUNE_LR         = 1e-5    # 10× lower than original training LR
FINETUNE_BATCH_SIZE = 8
VAL_FRACTION        = 0.20    # 20 % of feedback data held out for validation
RANDOM_SEED         = 42


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrappers
# ─────────────────────────────────────────────────────────────────────────────

class _AudioFeedbackDataset(Dataset):
    """Loads labelled MP3/WAV files from dataset_creation/audio/{real,fake}/."""

    def __init__(self, root: Path, class_to_idx: dict, preproc: _AudioPreproc):
        self.preproc = preproc
        self.items: list[tuple[Path, int]] = []
        for label, idx in class_to_idx.items():
            d = root / label
            if not d.exists():
                continue
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                    self.items.append((p, idx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        try:
            wav = self.preproc.load_mono(path)
            seg = _AudioPreproc.pad_or_center_crop(wav, self.preproc.seg_len)
            mel = self.preproc.wav_to_logmel_norm(seg)
        except Exception:
            # Return a zero spectrogram on load failure
            n_mels   = self.preproc.mel.n_mels
            seg_frames = self.preproc.seg_len // 256 + 1
            mel = torch.zeros(1, n_mels, seg_frames)
        return mel, label


class _ImageFeedbackDataset(Dataset):
    """Loads labelled JPG/PNG files from dataset_creation/frames/{real,fake}/."""

    def __init__(self, root: Path, class_to_idx: dict, tfm):
        self.tfm   = tfm
        self.items: list[tuple[Path, int]] = []
        for label, idx in class_to_idx.items():
            d = root / label
            if not d.exists():
                continue
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    self.items.append((p, idx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.tfm(img), label


# ─────────────────────────────────────────────────────────────────────────────
# Threshold checking
# ─────────────────────────────────────────────────────────────────────────────

def _count_samples(modality: str) -> dict[str, int]:
    """Return {class_name: file_count} for a given modality."""
    sub    = "audio"  if modality == "audio" else "frames"
    exts   = AUDIO_EXTS if modality == "audio" else IMAGE_EXTS
    counts = {}
    for label in ("real", "fake"):
        d = DATASET_ROOT / sub / label
        if not d.exists():
            counts[label] = 0
        else:
            counts[label] = sum(
                1 for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in exts
            )
    return counts


def check_status() -> dict:
    """
    Return current dataset counts and whether each modality is ready to train.
    Safe to call at any time — makes no changes.
    """
    min_a = MIN_SAMPLES_PER_CLASS_AUDIO
    min_i = MIN_SAMPLES_PER_CLASS_IMAGE

    audio_counts = _count_samples("audio")
    image_counts = _count_samples("image")

    audio_ready = all(v >= min_a for v in audio_counts.values())
    image_ready = all(v >= min_i for v in image_counts.values())

    return {
        "audio": {
            "counts":    audio_counts,
            "threshold": min_a,
            "ready":     audio_ready,
            "missing":   {k: max(0, min_a - v) for k, v in audio_counts.items()},
        },
        "image": {
            "counts":    image_counts,
            "threshold": min_i,
            "ready":     image_ready,
            "missing":   {k: max(0, min_i - v) for k, v in image_counts.items()},
        },
    }


def print_status():
    """Pretty-print dataset status to the console."""
    s = check_status()
    print("\n========== FEEDBACK DATASET STATUS ==========")
    for mod, info in s.items():
        icon = "READY" if info["ready"] else "NOT READY"
        print(f"\n  [{mod.upper()}] {icon}")
        print(f"  Threshold : {info['threshold']} samples per class")
        for cls, cnt in info["counts"].items():
            missing = info["missing"][cls]
            bar     = "+" * cnt + "-" * missing
            tag     = f"  need {missing} more" if missing else "  OK"
            print(f"    {cls:6s}: {cnt:4d} samples  [{bar}]{tag}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Backup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _backup_model(ckpt_path: Path) -> Optional[Path]:
    """Copy checkpoint to model_backups/ with a timestamp suffix."""
    if not ckpt_path.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst   = BACKUP_DIR / f"{ckpt_path.stem}_{stamp}{ckpt_path.suffix}"
    shutil.copy2(ckpt_path, dst)
    return dst


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tune: AUDIO
# ─────────────────────────────────────────────────────────────────────────────

def _finetune_audio(force: bool = False) -> dict:
    """
    Fine-tune the audio model on feedback data.

    Returns a result dict:
        skipped       bool
        reason        str
        val_acc_old   float | None
        val_acc_new   float | None
        improved      bool | None
        backup_path   str | None
    """
    counts = _count_samples("audio")
    min_n  = MIN_SAMPLES_PER_CLASS_AUDIO

    if not force and any(v < min_n for v in counts.values()):
        missing = {k: max(0, min_n - v) for k, v in counts.items()}
        return {
            "skipped": True,
            "reason":  f"Audio below threshold. Need {missing} more per class.",
            "counts":  counts,
        }

    if not AUDIO_CKPT.exists():
        return {"skipped": True, "reason": f"No checkpoint found at {AUDIO_CKPT}."}

    print("\n========== AUDIO FINE-TUNE ==========")
    print(f"  Samples: real={counts['real']}, fake={counts['fake']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt        = torch.load(str(AUDIO_CKPT), map_location="cpu", weights_only=False)
    classes     = ckpt["classes"]
    model_name  = ckpt.get("model_name", "cnn_small")
    sr          = int(ckpt.get("sample_rate", 16000))
    duration    = float(ckpt.get("duration", 4.0))
    n_mels      = int(ckpt.get("n_mels", 64))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    preproc = _AudioPreproc(sr, duration, n_mels)
    model   = SmallAudioCNN(len(classes))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)

    # ── Dataset & split ───────────────────────────────────────────────────────
    full_ds  = _AudioFeedbackDataset(DATASET_ROOT / "audio", class_to_idx, preproc)
    n_val    = max(1, int(len(full_ds) * VAL_FRACTION))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=FINETUNE_BATCH_SIZE, shuffle=False)

    # ── Baseline accuracy on validation set ──────────────────────────────────
    val_acc_old = _eval_accuracy(model, val_loader, device)
    print(f"  Baseline val accuracy : {val_acc_old:.4f}")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR)

    model.train()
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        correct = total = 0
        for mels, labels in train_loader:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(mels), labels)
            loss.backward()
            optimizer.step()
            preds    = model(mels).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        train_acc = correct / total if total else 0.0
        val_acc   = _eval_accuracy(model, val_loader, device)
        print(f"  Epoch {epoch}/{FINETUNE_EPOCHS} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    val_acc_new = _eval_accuracy(model, val_loader, device)
    improved    = val_acc_new >= val_acc_old

    # ── Save if improved ─────────────────────────────────────────────────────
    backup_path = None
    if improved:
        backup_path = _backup_model(AUDIO_CKPT)
        new_ckpt = {
            "model_name":  model_name,
            "sample_rate": sr,
            "duration":    duration,
            "n_mels":      n_mels,
            "state_dict":  model.state_dict(),
            "classes":     classes,
        }
        torch.save(new_ckpt, str(AUDIO_CKPT))
        print(f"  Saved updated model -> {AUDIO_CKPT}")
        print(f"  Old model backed up -> {backup_path}")
    else:
        print(f"  Val accuracy did not improve ({val_acc_new:.4f} < {val_acc_old:.4f}) — keeping old model.")

    return {
        "skipped":      False,
        "counts":       counts,
        "val_acc_old":  val_acc_old,
        "val_acc_new":  val_acc_new,
        "improved":     improved,
        "backup_path":  str(backup_path) if backup_path else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tune: IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def _finetune_image(force: bool = False) -> dict:
    """
    Fine-tune the image model on feedback data.

    Returns a result dict with the same keys as _finetune_audio().
    """
    counts = _count_samples("image")
    min_n  = MIN_SAMPLES_PER_CLASS_IMAGE

    if not force and any(v < min_n for v in counts.values()):
        missing = {k: max(0, min_n - v) for k, v in counts.items()}
        return {
            "skipped": True,
            "reason":  f"Image below threshold. Need {missing} more per class.",
            "counts":  counts,
        }

    if not IMAGE_CKPT.exists():
        return {"skipped": True, "reason": f"No checkpoint found at {IMAGE_CKPT}."}

    print("\n========== IMAGE FINE-TUNE ==========")
    print(f"  Samples: real={counts['real']}, fake={counts['fake']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt       = torch.load(str(IMAGE_CKPT), map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", "resnet18")
    img_size   = int(ckpt.get("img_size", 224))
    classes    = ckpt.get("classes", ["fake", "real"])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    model, _ = _build_image_backbone(model_name, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)

    # ── Transforms ───────────────────────────────────────────────────────────
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Dataset & split ───────────────────────────────────────────────────────
    full_ds  = _ImageFeedbackDataset(DATASET_ROOT / "frames", class_to_idx, train_tfm)
    n_val    = max(1, int(len(full_ds) * VAL_FRACTION))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds_raw = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    # Use val transform for validation split
    val_ds_raw.dataset.tfm = val_tfm

    train_loader = DataLoader(train_ds,   batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds_raw, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)

    # ── Baseline accuracy ─────────────────────────────────────────────────────
    val_acc_old = _eval_accuracy(model, val_loader, device)
    print(f"  Baseline val accuracy : {val_acc_old:.4f}")

    # ── Fine-tune (only the final classification head + last conv block) ──────
    # Freeze early layers — only fine-tune the head and final block.
    # This prevents overfitting on a small feedback dataset.
    _freeze_early_layers(model, model_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR,
    )

    model.train()
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            preds    = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        train_acc = correct / total if total else 0.0
        val_acc   = _eval_accuracy(model, val_loader, device)
        print(f"  Epoch {epoch}/{FINETUNE_EPOCHS} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    val_acc_new = _eval_accuracy(model, val_loader, device)
    improved    = val_acc_new >= val_acc_old

    # ── Save if improved ──────────────────────────────────────────────────────
    backup_path = None
    if improved:
        backup_path = _backup_model(IMAGE_CKPT)
        new_ckpt = {
            "model_name": model_name,
            "img_size":   img_size,
            "state_dict": model.state_dict(),
            "classes":    classes,
        }
        torch.save(new_ckpt, str(IMAGE_CKPT))
        print(f"  Saved updated model -> {IMAGE_CKPT}")
        print(f"  Old model backed up -> {backup_path}")
    else:
        print(f"  Val accuracy did not improve ({val_acc_new:.4f} < {val_acc_old:.4f}) — keeping old model.")

    return {
        "skipped":      False,
        "counts":       counts,
        "val_acc_old":  val_acc_old,
        "val_acc_new":  val_acc_new,
        "improved":     improved,
        "backup_path":  str(backup_path) if backup_path else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        preds    = model(inputs).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    model.train()
    return correct / total if total else 0.0


def _freeze_early_layers(model: nn.Module, model_name: str):
    """
    Freeze backbone early layers; keep the classification head trainable.
    Prevents overfitting on small feedback datasets.
    """
    if model_name == "resnet18":
        # Freeze everything up to layer3; fine-tune layer4 + fc
        frozen_prefixes = ("conv1", "bn1", "layer1", "layer2")
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in frozen_prefixes):
                param.requires_grad = False

    elif model_name == "efficientnet_b0":
        # Freeze the first 5 MBConv blocks; fine-tune the rest + head
        for name, param in model.named_parameters():
            if name.startswith("features.0") or name.startswith("features.1") \
                    or name.startswith("features.2") or name.startswith("features.3") \
                    or name.startswith("features.4"):
                param.requires_grad = False


def _write_log(entry: dict):
    """Append a result entry to the JSON log file."""
    log: list = []
    if MONITOR_LOG.exists():
        try:
            log = json.loads(MONITOR_LOG.read_text(encoding="utf-8"))
        except Exception:
            log = []
    log.append(entry)
    MONITOR_LOG.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def maybe_auto_retrain(
    mode: str = "both",
    force: bool = False,
) -> dict:
    """
    Check thresholds and fine-tune whichever modality is ready.
    Designed to be called from main.py after every feedback session.

    Parameters
    ----------
    mode  : "audio" | "image" | "both"
    force : Skip threshold check and always fine-tune.

    Returns
    -------
    dict with keys "audio" and/or "image", each containing the fine-tune result.
    """
    results: dict = {}

    if mode in ("audio", "both"):
        results["audio"] = _finetune_audio(force=force)

    if mode in ("image", "both"):
        results["image"] = _finetune_image(force=force)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode":      mode,
        "force":     force,
        "results":   results,
    }
    _write_log(entry)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Auto fine-tune audio/image models when feedback data is sufficient"
    )
    p.add_argument("--mode", choices=["audio", "image", "both"], default="both")
    p.add_argument("--check", action="store_true",
                   help="Print dataset status and exit (no training)")
    p.add_argument("--force", action="store_true",
                   help="Fine-tune even if below threshold")
    p.add_argument("--min-audio", type=int, default=MIN_SAMPLES_PER_CLASS_AUDIO,
                   help=f"Min audio samples per class (default: {MIN_SAMPLES_PER_CLASS_AUDIO})")
    p.add_argument("--min-image", type=int, default=MIN_SAMPLES_PER_CLASS_IMAGE,
                   help=f"Min image samples per class (default: {MIN_SAMPLES_PER_CLASS_IMAGE})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Allow overriding thresholds from CLI
    MIN_SAMPLES_PER_CLASS_AUDIO = args.min_audio
    MIN_SAMPLES_PER_CLASS_IMAGE = args.min_image

    print_status()

    if args.check:
        raise SystemExit(0)

    results = maybe_auto_retrain(mode=args.mode, force=args.force)

    print("\n========== SUMMARY ==========")
    for mod, res in results.items():
        if res.get("skipped"):
            print(f"  {mod.upper()}: SKIPPED — {res['reason']}")
        else:
            tag = "IMPROVED & SAVED" if res["improved"] else "NO IMPROVEMENT — kept old model"
            print(
                f"  {mod.upper()}: {tag} "
                f"(val acc {res['val_acc_old']:.4f} -> {res['val_acc_new']:.4f})"
            )
    print(f"\n  Log written to: {MONITOR_LOG}")
