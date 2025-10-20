# minimal_infer_by_path.py
import json
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# --------- SET THIS ---------
AUDIO_PATH   = r"test.mp3"  # file or folder
CKPT_PATH    = r"best_audio_model.pt"          # your saved checkpoint
LABELS_JSON  = r"class_to_idx_audio.json"      # optional (used only to warn on mismatch)
DEVICE_PREF  = "cuda"                          # "cuda" or "cpu"
TOPK         = 3
# ----------------------------

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# ---- model (same as training) ----
class SmallAudioCNN(nn.Module):
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
        return self.head(self.net(x))

def build_model(num_classes: int, backbone: str):
    if backbone != "cnn_small":
        raise ValueError(f"Unsupported backbone in ckpt: {backbone}")
    return SmallAudioCNN(num_classes)

# ---- preprocessing (mirror training) ----
class Preprocessor:
    def __init__(self, sample_rate: int, duration: float, n_mels: int):
        self.sr = sample_rate
        self.seg_len = int(sample_rate * duration)
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024, hop_length=256, win_length=1024,
            n_mels=n_mels, f_min=20.0, f_max=sample_rate // 2
        )
        self.to_db = AmplitudeToDB(stype="power")
        self._resamplers = {}

    def _resampler(self, from_sr):
        if from_sr == self.sr:
            return None
        if from_sr not in self._resamplers:
            self._resamplers[from_sr] = torchaudio.transforms.Resample(from_sr, self.sr)
        return self._resamplers[from_sr]

    def load_mono(self, path: Path):
        wav, sr = torchaudio.load(str(path))  # (C, T)
        wav = wav.mean(dim=0) if wav.ndim == 2 else wav.squeeze(0)
        rs = self._resampler(sr)
        return rs(wav) if rs else wav  # (T,)

    @staticmethod
    def _pad_or_center_crop(wav, target_len):
        T = wav.numel()
        if T == target_len:
            return wav
        if T > target_len:
            start = (T - target_len) // 2
            return wav[start:start+target_len]
        out = torch.zeros(target_len, dtype=wav.dtype)
        out[:T] = wav
        return out

    def wav_to_logmel_norm(self, wav_1d: torch.Tensor):
        mel = self.mel(wav_1d.unsqueeze(0))   # (1, n_mels, Tm)
        mel_db = self.to_db(mel)
        m, s = mel_db.mean(), mel_db.std()
        return (mel_db - m) / (s + 1e-6)      # (1, n_mels, Tm)

# ---- utility ----
def list_audio_files(p: Path):
    if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
        return [p]
    return sorted([x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in AUDIO_EXTS])

# ---- load checkpoint & prepare model ----
ckpt = torch.load(CKPT_PATH, map_location="cpu")
backbone = ckpt.get("model_name", "cnn_small")
sr       = int(ckpt.get("sample_rate", 16000))
duration = float(ckpt.get("duration", 4.0))
n_mels   = int(ckpt.get("n_mels", 64))
classes  = ckpt["classes"]
num_classes = len(classes)

device = torch.device(DEVICE_PREF if (DEVICE_PREF == "cuda" and torch.cuda.is_available()) else "cpu")
print(f"Device: {device} | Classes: {classes}")

# optional sanity check with JSON
try:
    with open(LABELS_JSON, "r") as f:
        ctij = json.load(f)  # {name: idx}
    warn = any((name not in ctij) or (ctij[name] != i) for i, name in enumerate(classes))
    if warn:
        print("[WARN] class_to_idx_audio.json order differs from ckpt. Using ckpt order.")
except Exception:
    pass

model = build_model(num_classes, backbone)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.to(device).eval()

pre = Preprocessor(sr, duration, n_mels)

@torch.no_grad()
def predict_audio_path(path: str, hop_sec: float = 2.0):
    """
    Give me a file path. If it's longer than `duration`, we do sliding windows and average logits.
    Returns: (pred_label, pred_prob, probs_dict)
    """
    p = Path(path)
    wav = pre.load_mono(p)
    seg_len = int(sr * duration)
    hop_len = int(sr * hop_sec) if hop_sec else seg_len

    # single short clip
    if wav.numel() <= seg_len:
        seg = pre._pad_or_center_crop(wav, seg_len)
        mel = pre.wav_to_logmel_norm(seg).unsqueeze(0).to(device)  # (1,1,n_mels,Tm)
        logits = model(mel).squeeze(0)
    else:
        # sliding windows with mean logits
        start, n, logits_sum = 0, 0, None
        T = wav.numel()
        while start < T:
            end = min(start + seg_len, T)
            chunk = wav[start:end]
            if chunk.numel() < seg_len:
                chunk = pre._pad_or_center_crop(chunk, seg_len)
            mel = pre.wav_to_logmel_norm(chunk).unsqueeze(0).to(device)
            lg = model(mel).squeeze(0)  # (C,)
            logits_sum = lg if logits_sum is None else (logits_sum + lg)
            n += 1
            if end == T:
                break
            start += hop_len
        logits = logits_sum / max(n, 1)

    probs = torch.softmax(logits, dim=-1).cpu()
    idx = int(torch.argmax(probs).item())
    pred_label = classes[idx]
    pred_prob = float(probs[idx].item())
    probs_dict = {cls: float(probs[i].item()) for i, cls in enumerate(classes)}
    return pred_label, pred_prob, probs_dict

# --------- RUN on your path (file or folder) ---------
target = Path(AUDIO_PATH)
files = list_audio_files(target) if target.exists() else []
if not files:
    raise FileNotFoundError(f"No audio found at {AUDIO_PATH}")

for fp in files:
    label, p, dist = predict_audio_path(str(fp))
    # pretty print top-k
    import heapq
    top = heapq.nlargest(min(TOPK, len(dist)), dist.items(), key=lambda x: x[1])
    top_str = ", ".join([f"{k}={v:.3f}" for k, v in top])
    print(f"[{fp.name}] → {label} ({p:.3f}) | top-{len(top)}: {top_str}")
