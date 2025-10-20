#!/usr/bin/env python3
# Train.py — Audio training with robust split discovery + per-run results folder
import argparse
import csv
import json
import os
import signal
import sys
import time
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, TimeMasking, FrequencyMasking

# Optional metrics (ROC-AUC, AUPRC). If not installed, we fall back gracefully.
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# --------------------- quiet & deterministic-ish defaults ---------------------
warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# --------------------- logging helper ---------------------
class Logger:
    """
    Tee-style logger: writes to file and stdout. Flushes on every call.
    """
    def __init__(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filepath, "w", buffering=1)  # line buffered
        self.start_time = time.time()
        self._log_header(filepath)

    def _log_header(self, filepath):
        self.log("=" * 80)
        self.log(f"Log file: {Path(filepath).resolve()}")
        self.log(f"Start time: {datetime.now().isoformat()}")
        self.log("=" * 80)

    def log(self, msg: str = ""):
        print(msg)
        try:
            self.file.write(msg + "\n")
            self.file.flush()
            os.fsync(self.file.fileno())
        except Exception:
            # If disk/permission error, at least keep console output going.
            pass

    def close(self):
        elapsed = time.time() - self.start_time
        self.log(f"\nTotal elapsed: {elapsed:.2f}s")
        self.log("=" * 80)
        try:
            self.file.close()
        except Exception:
            pass

# Global logger handle (set in train())
logger: Optional[Logger] = None

def safe_log(msg: str):
    if logger:
        logger.log(msg)
    else:
        print(msg)

# --------------------- metrics helpers ---------------------
def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def precision_recall_f1_from_cm(cm):
    precisions, recalls, f1s = [], [], []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2*precision*recall / (precision+recall)) if (precision+recall) > 0 else 0.0
        precisions.append(precision); recalls.append(recall); f1s.append(f1)
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return precisions, recalls, f1s, macro_p, macro_r, macro_f1

# --------------------- audio utils ---------------------
def list_audio_files(root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """
    Expects root to contain subfolders per class (e.g., fake/, real/).
    Returns list of (filepath, class_index) and classes list.
    """
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    idx_map = {c: i for i, c in enumerate(classes)}
    items = []
    for cls in classes:
        cdir = root / cls
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                items.append((p, idx_map[cls]))
    return items, classes

def pad_or_trim(waveform: torch.Tensor, target_len: int, train_mode: bool) -> torch.Tensor:
    """
    waveform: (T,) mono
    If longer than target, random crop in train, center crop in eval.
    If shorter, pad with zeros at the end.
    """
    T = waveform.shape[-1]
    if T == target_len:
        return waveform
    if T > target_len:
        if train_mode:
            start = random.randint(0, T - target_len)
        else:
            start = (T - target_len) // 2
        return waveform[start:start+target_len]
    # pad
    out = torch.zeros(target_len, dtype=waveform.dtype)
    out[:T] = waveform
    return out

# --------------------- dataset ---------------------
class SafeAudioDataset(Dataset):
    """
    Robust audio dataset:
    - loads mono, resamples, pads/crops to fixed length
    - returns log-mel spectrogram (C=1, F=n_mels, T=time)
    - applies SpecAugment (train only)
    - on load failure, returns a zero spectrogram with correct shape
    """
    def __init__(self,
                 split_root: Path,
                 sample_rate: int = 16000,
                 duration_sec: float = 4.0,
                 n_mels: int = 64,
                 train: bool = False):
        self.items, self.classes = list_audio_files(split_root)
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration_sec)
        self.train = train

        # pre-define transforms
        self.resampler = None  # create lazily if needed
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024, hop_length=256, win_length=1024,
            n_mels=n_mels, f_min=20.0, f_max=sample_rate // 2
        )
        self.to_db = AmplitudeToDB(stype="power")

        # SpecAugment for training
        self.spec_time_mask = TimeMasking(time_mask_param=24) if train else None
        self.spec_freq_mask = FrequencyMasking(freq_mask_param=8) if train else None

    def __len__(self):
        return len(self.items)

    def _safe_load(self, path: Path) -> Tuple[torch.Tensor, int]:
        try:
            wav, sr = torchaudio.load(str(path))  # (C, T)
            if wav.ndim == 2:
                wav = wav.mean(dim=0)  # mono
            else:
                wav = wav.squeeze(0)
            if sr != self.sample_rate:
                if self.resampler is None:
                    self.resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                wav = self.resampler(wav)
            return wav, self.sample_rate
        except Exception as e:
            safe_log(f"[WARN] Failed to load {path}: {e}. Using zero audio.")
            return torch.zeros(self.target_len), self.sample_rate

    def __getitem__(self, index):
        path, target = self.items[index]
        wav, _ = self._safe_load(path)
        wav = pad_or_trim(wav, self.target_len, self.train)

        # (T,) -> (1, T) for mel
        mel = self.mel(wav.unsqueeze(0))   # (1, n_mels, time)
        mel_db = self.to_db(mel)           # log-mel

        # normalize per-sample (mean/var)
        m = mel_db.mean()
        s = mel_db.std()
        mel_db = (mel_db - m) / (s + 1e-6)

        if self.train:
            if self.spec_freq_mask is not None:
                mel_db = self.spec_freq_mask(mel_db)
            if self.spec_time_mask is not None:
                mel_db = self.spec_time_mask(mel_db)

        return mel_db, target

# --------------------- model ---------------------
class SmallAudioCNN(nn.Module):
    """
    Simple but strong CNN for (1, n_mels, T) log-mels.
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveMaxPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.net(x)   # (B, 256, 4, 4)
        x = self.head(x)  # (B, n_classes)
        return x

def build_model(num_classes: int, backbone: str):
    if backbone == "cnn_small":
        return SmallAudioCNN(num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

# --------------------- evaluation ---------------------
@torch.no_grad()
def evaluate(model, loader, device, criterion=None, need_probs: bool = False):
    model.eval()
    loss_sum, total, correct = 0.0, 0, 0
    y_true, y_pred = [], []
    y_scores = []  # softmax prob for class 1 if binary

    for mels, labels in loader:
        mels = mels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(mels)
        if criterion is not None:
            loss = criterion(logits, labels)
            loss_sum += loss.item() * mels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        if need_probs:
            if logits.size(1) == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]  # prob of class index 1
                y_scores.extend(probs.detach().cpu().numpy().tolist())

    acc = correct / total if total else 0.0
    avg_loss = loss_sum / total if (criterion is not None and total) else None

    if need_probs and logits.size(1) == 2:
        return avg_loss, acc, np.array(y_true), np.array(y_pred), np.array(y_scores)
    else:
        return avg_loss, acc, np.array(y_true), np.array(y_pred), None

# --------------------- robust split discovery ---------------------
def _resolve_split(root: Path, primary: str, fallback: str) -> Path:
    """
    Improved resolver: Automatically finds train/val/test folders even if nested
    or named differently (training, validation, testing).
    """
    synonyms = {
        "train": ["train", "training"],
        "val":   ["val", "validation"],
        "test":  ["test", "testing"],
    }

    # 1) direct children first
    for name in synonyms.get(primary, [primary]) + [fallback]:
        p = root / name
        if p.exists():
            return p

    # 2) look one level deeper
    for child in root.iterdir():
        if not child.is_dir():
            continue
        for name in synonyms.get(primary, [primary]) + [fallback]:
            p = child / name
            if p.exists():
                return p

    # 3) recursive walk (last resort)
    for dirpath, dirnames, _ in os.walk(root):
        for name in synonyms.get(primary, [primary]) + [fallback]:
            if name in dirnames:
                return Path(dirpath) / name

    raise FileNotFoundError(f"Could not find split '{primary}' or '{fallback}' under {root}")

def load_datasets(data_root: Path, sample_rate: int, duration: float, n_mels: int):
    train_root = _resolve_split(data_root, "train", "training")
    val_root   = _resolve_split(data_root, "val", "validation")
    test_root  = _resolve_split(data_root, "test", "testing")

    train_ds = SafeAudioDataset(train_root, sample_rate, duration, n_mels, train=True)
    val_ds   = SafeAudioDataset(val_root,   sample_rate, duration, n_mels, train=False)
    test_ds  = SafeAudioDataset(test_root,  sample_rate, duration, n_mels, train=False)

    # persist mapping (class name -> index)
    class_to_idx = {c: i for i, c in enumerate(train_ds.classes)}
    with open("class_to_idx_audio.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    # dataset stats & resolved paths
    safe_log(f"Classes: {train_ds.classes}")
    safe_log(f"[DATASET] ✅ train root: {train_root}")
    safe_log(f"[DATASET] ✅ val root:   {val_root}")
    safe_log(f"[DATASET] ✅ test root:  {test_root}")
    safe_log(f"Training samples: {len(train_ds)} | Validation: {len(val_ds)} | Testing: {len(test_ds)}")

    return train_ds, val_ds, test_ds

# --------------------- CSV/JSON history helpers ---------------------
class HistoryWriter:
    def __init__(self, csv_path: Path, json_path: Path):
        self.csv_path = csv_path
        self.json_path = json_path
        self.rows = []
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Init CSV with header
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "note"])

    def add(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, note=""):
        self.rows.append({
            "epoch": epoch,
            "train_loss": float(train_loss) if train_loss is not None else None,
            "train_acc": float(train_acc) if train_acc is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "val_acc": float(val_acc) if val_acc is not None else None,
            "lr": float(lr) if lr is not None else None,
            "note": note
        })
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr, note])

    def finalize(self, metadata: dict):
        payload = {"metadata": metadata, "history": self.rows}
        with open(self.json_path, "w") as f:
            json.dump(payload, f, indent=2)

# --------------------- training ---------------------
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _setup_interrupt_handler(state, out_dir: Path):
    def handler(signum, frame):
        safe_log("\n[INTERRUPT] Received signal, attempting graceful shutdown...")
        # save last checkpoint if possible
        try:
            torch.save(state(), out_dir / "last_interrupt_checkpoint.pt")
            safe_log(f"Saved '{out_dir / 'last_interrupt_checkpoint.pt'}'")
        except Exception as e:
            safe_log(f"Failed to save interrupt checkpoint: {e}")
        finally:
            if logger:
                logger.close()
            sys.exit(130)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

def train(args):
    global logger

    # ------------- results folder per run -------------
    run_dir = Path(args.results_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------- logging -------------
    logger = Logger(run_dir / "audio_training_results.txt")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # ------------- device & args -------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_log(f"Device: {device} | PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        safe_log(f"GPU name: {torch.cuda.get_device_name(0)}")
    safe_log(f"Args: {vars(args)}")
    safe_log(f"[RESULTS DIR] {run_dir.resolve()}")

    # ------------- data -------------
    data_root = Path(args.data_root).resolve()
    safe_log(f"[DATA ROOT arg] {data_root}")
    # Quick visibility check
    safe_log(f"[CHECK] train: {(data_root/'train').exists()} | training: {(data_root/'training').exists()}")
    safe_log(f"[CHECK] val:   {(data_root/'val').exists()}   | validation: {(data_root/'validation').exists()}")
    safe_log(f"[CHECK] test:  {(data_root/'test').exists()}  | testing: {(data_root/'testing').exists()}")

    train_ds, val_ds, test_ds = load_datasets(data_root, args.sample_rate, args.duration, args.n_mels)

    pw = bool(args.workers > 0)
    pin = (device.type == "cuda")
    prefetch = 2 if pw else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin,
                              persistent_workers=pw, prefetch_factor=prefetch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin,
                            persistent_workers=pw, prefetch_factor=prefetch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=pin,
                             persistent_workers=pw, prefetch_factor=prefetch)

    # ------------- model/opt/sched -------------
    num_classes = len(train_ds.classes)
    model = build_model(num_classes, args.backbone).to(device)
    n_params = count_trainable_parameters(model)
    safe_log(f"Model: {args.backbone} | Trainable params: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda") and (not args.no_amp))

    # ------------- history writers -------------
    hist_csv = run_dir / "training_history.csv"
    hist_json = run_dir / "training_history.json"
    history = HistoryWriter(csv_path=hist_csv, json_path=hist_json)

    # ------------- early stop bookkeeping -------------
    best_val_acc, patience_left, best_epoch = 0.0, args.early_stop, -1

    # ------------- interrupt-safe checkpoint lambda -------------
    def current_state():
        return {
            "model_name": args.backbone,
            "sample_rate": args.sample_rate,
            "duration": args.duration,
            "n_mels": args.n_mels,
            "state_dict": model.state_dict(),
            "classes": train_ds.classes,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch
        }
    _setup_interrupt_handler(current_state, run_dir)

    # ------------- training loop -------------
    wall_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        epoch_start = time.time()

        for mels, labels in train_loader:
            mels = mels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda") and (not args.no_amp)):
                logits = model(mels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item() * mels.size(0)
            run_correct += (logits.argmax(1) == labels).sum().item()
            run_total += labels.size(0)

        train_loss = run_loss / run_total if run_total else 0.0
        train_acc = run_correct / run_total if run_total else 0.0

        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, device, criterion, need_probs=False)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start
        msg = (f"Epoch {epoch:02d}/{args.epochs} | "
               f"Train {train_loss:.4f}/{train_acc:.4f} | "
               f"Val {val_loss:.4f}/{val_acc:.4f} | "
               f"LR {current_lr:.2e} | time {epoch_time:.1f}s")
        safe_log(msg)
        history.add(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, note="")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(current_state(), run_dir / args.output)
            safe_log(f"✅ Saved BEST to {run_dir / args.output} (best_val_acc={best_val_acc:.4f}, epoch={best_epoch})")
            patience_left = args.early_stop
        else:
            patience_left -= 1
            safe_log(f"No improvement. Early-stop patience left: {patience_left}")
            if patience_left == 0:
                safe_log("⛔ Early stopping triggered.")
                break

    # ------------- rollback to best & test -------------
    safe_log("\nRolling back to best checkpoint for final evaluation...")
    ckpt_path = run_dir / args.output
    ckpt = torch.load(ckpt_path, map_location="cpu")
    rolled_epoch = ckpt.get("best_epoch", "unknown")
    safe_log(f"Rolled back to epoch: {rolled_epoch}, best_val_acc={ckpt.get('best_val_acc', None)}")

    model = build_model(num_classes, args.backbone).to(device)
    model.load_state_dict(ckpt["state_dict"])

    # Evaluate with probabilities if binary to compute ROC/AUPRC
    need_probs = (num_classes == 2)
    test_loss, test_acc, y_true, y_pred, y_scores = evaluate(
        model, test_loader, device, criterion, need_probs=need_probs
    )
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    per_p, per_r, per_f1, mp, mr, mf1 = precision_recall_f1_from_cm(cm)

    safe_log("\n=== TEST RESULTS ===")
    safe_log(f"Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
    safe_log(f"Macro Precision: {mp:.4f} | Macro Recall: {mr:.4f} | Macro F1: {mf1:.4f}")
    for i, cls in enumerate(train_ds.classes):
        safe_log(f"Class '{cls}': P={per_p[i]:.4f} R={per_r[i]:.4f} F1={per_f1[i]:.4f}")
    safe_log("\nConfusion matrix (rows=true, cols=pred):\n" + str(cm))

    # Extra metrics for binary classification if sklearn available
    extra_metrics = {}
    if need_probs and (y_scores is not None) and SKLEARN_OK:
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            extra_metrics = {"roc_auc": float(roc_auc), "average_precision": float(ap)}
            safe_log(f"\nExtra metrics (binary): ROC-AUC={roc_auc:.4f} | Average Precision (AUPRC)={ap:.4f}")
        except Exception as e:
            safe_log(f"[WARN] Could not compute ROC/AUPRC: {e}")
    elif need_probs and not SKLEARN_OK:
        safe_log("[INFO] sklearn not available; skipping ROC-AUC and AUPRC.")

    # ------------- finalize JSON history with metadata -------------
    metadata = {
        "datetime_start": datetime.fromtimestamp(wall_start).isoformat(),
        "datetime_end": datetime.now().isoformat(),
        "wall_clock_total_s": time.time() - wall_start,
        "args": vars(args),
        "device": str(device),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "classes": train_ds.classes,
        "num_classes": num_classes,
        "best_epoch": rolled_epoch,
        "best_val_acc": float(ckpt.get("best_val_acc", 0.0)),
        "final_test": {
            "loss": float(test_loss) if test_loss is not None else None,
            "accuracy": float(test_acc),
            "macro_precision": float(mp),
            "macro_recall": float(mr),
            "macro_f1": float(mf1),
            "per_class_precision": [float(x) for x in per_p],
            "per_class_recall": [float(x) for x in per_r],
            "per_class_f1": [float(x) for x in per_f1],
            "confusion_matrix": cm.tolist(),
            **extra_metrics
        }
    }
    history.finalize(metadata)
    safe_log(f"\nSaved per-epoch CSV: {hist_csv}")
    safe_log(f"Saved full history JSON: {hist_json}")
    safe_log(f"Saved class map JSON: {Path('class_to_idx_audio.json').resolve()}")
    safe_log(f"Saved best checkpoint: {ckpt_path.resolve()}")

    if logger:
        logger.close()

# --------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train audio real vs fake classifier with robust split discovery + per-run results")
    p.add_argument("--data-root", type=str, default=".", help="Folder that (directly or nested) contains train|training, val|validation, test|testing")
    p.add_argument("--backbone", type=str, default="cnn_small", choices=["cnn_small"])
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=4.0, help="seconds per training sample (crop/pad)")
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stop", type=int, default=4, help="patience epochs without val acc improvement")
    p.add_argument("--output", type=str, default="best_audio_model.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP)")
    p.add_argument("--results-dir", type=str, default="results", help="Directory where per-run folders will be created")
    return p.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        train(args)
    except Exception as e:
        safe_log(f"\n[FATAL] Unhandled exception: {e}")
        raise
