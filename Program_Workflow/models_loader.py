# models_loader.py
"""
Importable module to load & run your deepfake AUDIO and IMAGE models.

Usage:
    from models_loader import (
        load_audio_model, predict_audio,
        load_image_model, predict_image
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a_mdl, a_meta = load_audio_model(device=device)
    i_mdl, i_meta = load_image_model(device=device)

    print(predict_audio(a_mdl, "clip.mp3", a_meta, topk=3))
    print(predict_image(i_mdl, "frame.jpg", i_meta, topk=3))
"""

from pathlib import Path
import json
import heapq
from typing import Dict, Tuple, List, Any

import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision import models, transforms
from PIL import Image, ImageFile

# robust image loading
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------------- AUDIO ---------------------------------------
# Matches your training/inference architecture & preprocessing:contentReference[oaicite:0]{index=0}
class SmallAudioCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.head(self.net(x))


class _AudioPreproc:
    """Mirror of your training mel-spectrogram + log-db + standardize pipeline:contentReference[oaicite:1]{index=1}."""
    def __init__(self, sample_rate: int, duration_s: float, n_mels: int):
        self.sr = int(sample_rate)
        self.seg_len = int(self.sr * float(duration_s))
        self.mel = MelSpectrogram(
            sample_rate=self.sr, n_fft=1024, hop_length=256, win_length=1024,
            n_mels=int(n_mels), f_min=20.0, f_max=self.sr // 2
        )
        self.to_db = AmplitudeToDB(stype="power")
        self._resamplers: Dict[int, Any] = {}

    def _resampler(self, from_sr: int):
        if from_sr == self.sr:
            return None
        if from_sr not in self._resamplers:
            self._resamplers[from_sr] = torchaudio.transforms.Resample(from_sr, self.sr)
        return self._resamplers[from_sr]

    def load_mono(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0) if wav.ndim == 2 else wav.squeeze(0)
        rs = self._resampler(int(sr))
        return rs(wav) if rs else wav

    @staticmethod
    def pad_or_center_crop(wav: torch.Tensor, target_len: int) -> torch.Tensor:
        T = wav.numel()
        if T == target_len:
            return wav
        if T > target_len:
            start = (T - target_len) // 2
            return wav[start:start + target_len]
        out = torch.zeros(target_len, dtype=wav.dtype)
        out[:T] = wav
        return out

    def wav_to_logmel_norm(self, wav_1d: torch.Tensor) -> torch.Tensor:
        mel = self.mel(wav_1d.unsqueeze(0))      # (1, n_mels, Tm)
        mel_db = self.to_db(mel)
        m, s = mel_db.mean(), mel_db.std()
        return (mel_db - m) / (s + 1e-6)         # (1, n_mels, Tm)


def load_audio_model(
    ckpt_path: str = "best_audio_model.pt",
    labels_json: str = "class_to_idx_audio.json",
    device: str = None,
):
    """
    Returns:
        model: torch.nn.Module (eval mode, on device)
        meta: dict with keys
              - device, classes(List[str]), idx_to_class(dict), preproc(_AudioPreproc),
                seg_len(int), hop_len_default(int)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes: List[str] = ckpt["classes"]
    model_name = ckpt.get("model_name", "cnn_small")
    sr = int(ckpt.get("sample_rate", 16000))
    duration = float(ckpt.get("duration", 4.0))
    n_mels = int(ckpt.get("n_mels", 64))

    if model_name != "cnn_small":
        raise ValueError(f"Unsupported audio backbone in ckpt: {model_name}")

    model = SmallAudioCNN(len(classes))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()

    # label map (you also have fake/real mapping in your JSONs):contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}
    with open(labels_json, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    preproc = _AudioPreproc(sr, duration, n_mels)
    meta = dict(
        device=device,
        classes=classes,
        idx_to_class=idx_to_class,
        preproc=preproc,
        seg_len=int(sr * duration),
        hop_len_default=int(sr * 2.0),  # default 2s hop like your reference script:contentReference[oaicite:5]{index=5}
    )
    return model, meta


@torch.no_grad()
def predict_audio(
    model: nn.Module,
    audio_path: str,
    meta: Dict[str, Any],
    hop_seconds: float = 2.0,
    topk: int = 3,
):
    """
    Predict on a single audio file. Uses sliding windows if longer than segment length.
    Returns dict: {"label": str, "prob": float, "topk": [(label, prob), ...], "dist": {label: prob}}
    """
    device = torch.device(meta["device"])
    pre = meta["preproc"]
    seg_len = meta["seg_len"]
    path = Path(audio_path)

    wav = pre.load_mono(path)
    hop_len = int(pre.sr * hop_seconds) if hop_seconds else seg_len

    # single segment
    if wav.numel() <= seg_len:
        seg = pre.pad_or_center_crop(wav, seg_len)
        mel = pre.wav_to_logmel_norm(seg).unsqueeze(0).to(device)  # (1,1,n_mels,Tm)
        logits = model(mel).squeeze(0)
    else:
        # mean logits across sliding windows:contentReference[oaicite:6]{index=6}
        start, n, logits_sum = 0, 0, None
        T = wav.numel()
        while start < T:
            end = min(start + seg_len, T)
            chunk = wav[start:end]
            if chunk.numel() < seg_len:
                chunk = pre.pad_or_center_crop(chunk, seg_len)
            mel = pre.wav_to_logmel_norm(chunk).unsqueeze(0).to(device)
            lg = model(mel).squeeze(0)
            logits_sum = lg if logits_sum is None else (logits_sum + lg)
            n += 1
            if end == T:
                break
            start += hop_len
        logits = logits_sum / max(n, 1)

    probs = torch.softmax(logits, dim=-1).cpu()
    dist = {meta["classes"][i]: float(probs[i].item()) for i in range(len(meta["classes"]))}
    pred_idx = int(torch.argmax(probs).item())
    pred_label = meta["classes"][pred_idx]
    pred_prob = float(probs[pred_idx].item())
    top = heapq.nlargest(min(topk, len(dist)), dist.items(), key=lambda x: x[1])
    return {"label": pred_label, "prob": pred_prob, "topk": top, "dist": dist}

# ------------------------------- IMAGE ---------------------------------------
# Reconstructs the CNN backbone & head exactly like your GUI loader:contentReference[oaicite:7]{index=7}
def _build_image_backbone(model_name: str, num_classes: int):
    if model_name == "resnet18":
        mdl = models.resnet18(weights=None)
        in_f = mdl.fc.in_features
        mdl.fc = nn.Linear(in_f, num_classes)
        last = "fc"
    elif model_name == "efficientnet_b0":
        mdl = models.efficientnet_b0(weights=None)
        in_f = mdl.classifier[1].in_features
        mdl.classifier[1] = nn.Linear(in_f, num_classes)
        last = "classifier.1"
    else:
        raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")
    return mdl, last


def load_image_model(
    ckpt_path: str = "best_model.pt",
    labels_json: str = "class_to_idx_img.json",
    device: str = None,
):
    """
    Returns:
        model: torch.nn.Module (eval mode, on device)
        meta: dict with keys
              - device, tfm(torchvision transform), idx_to_class(dict), classes(list), img_size(int)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt.get("model_name", "resnet18")
    img_size = int(ckpt.get("img_size", 224))
    state_dict = ckpt["state_dict"]

    # labels
    with open(labels_json, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    # backbone and head
    model, _ = _build_image_backbone(model_name, num_classes=len(idx_to_class))
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    # preprocessing exactly as your loader does (size + ImageNet norm):contentReference[oaicite:8]{index=8}
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    meta = dict(
        device=device,
        tfm=tfm,
        idx_to_class=idx_to_class,
        classes=classes,
        img_size=img_size,
        model_name=model_name,
    )
    return model, meta


@torch.no_grad()
def predict_image(
    model: nn.Module,
    image_path: str,
    meta: Dict[str, Any],
    topk: int = 3,
):
    """
    Predict on a single image file.
    Returns dict: {"label": str, "prob": float, "topk": [(label, prob), ...], "dist": {label: prob}}
    """
    device = torch.device(meta["device"])
    tfm = meta["tfm"]
    idx_to_class = meta["idx_to_class"]

    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    logits = model(x).squeeze(0)
    probs = torch.softmax(logits, dim=-1).cpu()

    dist = {idx_to_class[i]: float(probs[i].item()) for i in range(len(idx_to_class))}
    pred_idx = int(torch.argmax(probs).item())
    pred_label = idx_to_class[pred_idx]
    pred_prob = float(probs[pred_idx].item())
    top = heapq.nlargest(min(topk, len(dist)), dist.items(), key=lambda x: x[1])
    return {"label": pred_label, "prob": pred_prob, "topk": top, "dist": dist}

# ---- Add these to models_loader.py (below the existing code) -----------------

from datetime import datetime

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _gather_files(target, exts):
    p = Path(target)
    if p.is_file():
        return [p] if p.suffix.lower() in exts else []
    return sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in exts)


def _format_pred_lines(kind: str, path: Path, result: dict, sep="  "):
    """kind Ã¢Ë†Ë† {'AUDIO','IMAGE'}; result from predict_*()"""
    label = result["label"]
    prob  = result["prob"]
    lines = [
        f"[{kind}] {path}",
        f"{sep}Prediction: {label}",
        f"{sep}Confidence (prob): {prob:.4f}",
    ]
    if "topk" in result and result["topk"]:
        top_str = ", ".join([f"{k}={v:.3f}" for k, v in result["topk"]])
        lines.append(f"{sep}Top-{len(result['topk'])}: {top_str}")
    return "\n".join(lines)


def save_audio_results(
    model,
    meta,
    input_path_or_dir: str,
    out_txt: str = "audio_predictions.txt",
    hop_seconds: float = 2.0,
    topk: int = 3,
):
    """
    Print per-file AUDIO results (label + prob) to console AND save full details to a .txt file.
    Accepts a single file or a folder; recursively scans audio files.
    """
    paths = _gather_files(input_path_or_dir, _AUDIO_EXTS)
    if not paths:
        print(f"[AUDIO] No audio files found at: {input_path_or_dir}")
        return

    header = [
        f"# AUDIO PREDICTIONS",
        f"# Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"# Source: {Path(input_path_or_dir).resolve()}",
        "",
    ]
    blocks = []
    for p in paths:
        res = predict_audio(model, str(p), meta, hop_seconds=hop_seconds, topk=topk)
        # print brief result
        print(f"[AUDIO] {p.name} -> {res['label']} (prob={res['prob']:.4f})")
        # collect detailed block
        blocks.append(_format_pred_lines("AUDIO", p, res))

    out_path = Path(out_txt)
    out_path.write_text("\n".join(header + blocks), encoding="utf-8")
    print(f"[AUDIO] Saved full results to: {out_path.resolve()}")


def save_image_results(
    model,
    meta,
    input_path_or_dir: str,
    out_txt: str = "image_predictions.txt",
    topk: int = 3,
):
    """
    Print per-file IMAGE results (label + prob) to console AND save full details to a .txt file.
    Accepts a single file or a folder; recursively scans image files.
    """
    paths = _gather_files(input_path_or_dir, _IMAGE_EXTS)
    if not paths:
        print(f"[IMAGE] No image files found at: {input_path_or_dir}")
        return

    header = [
        f"# IMAGE PREDICTIONS",
        f"# Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"# Source: {Path(input_path_or_dir).resolve()}",
        "",
    ]
    blocks = []
    for p in paths:
        res = predict_image(model, str(p), meta, topk=topk)
        # print brief result
        print(f"[IMAGE] {p.name} -> {res['label']} (prob={res['prob']:.4f})")
        # collect detailed block
        blocks.append(_format_pred_lines("IMAGE", p, res))

    out_path = Path(out_txt)
    out_path.write_text("\n".join(header + blocks), encoding="utf-8")
    print(f"[IMAGE] Saved full results to: {out_path.resolve()}")
