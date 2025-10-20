# feedback.py â€” dataset_creation/{audio|frames}/{real|fake} with preview â†’ confirm flow
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime
import random
import shlex
import shutil
import subprocess

# Reuse your existing extractor and models
from video_io import split_video_to_mp3_and_frames
from models_loader import (
    load_audio_model, predict_audio,
    load_image_model, predict_image
)

@dataclass
class FBConfig:
    dataset_root: Path                # set to Path("dataset_creation")
    img_fps: float = 2.0
    img_ext: str = "jpg"
    jpg_quality: int = 90
    audio_bitrate: str = "192k"
    # video normalisation kept for robust audio extraction even if we don't store video
    normalize_video: bool = True
    video_height: int = 720
    video_fps: int = 30
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_sr: int = 16000
    audio_channels: int = 1
    # split removed (no train/val/test in this layout)
    seed: int = 42
    log_csv: str = "feedback_log.csv"

# ------------- utils -------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _ffprobe_ok(path: Path) -> bool:
    try:
        cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(str(path))}'
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        return proc.returncode == 0 and bool(proc.stdout.decode(errors="ignore").strip())
    except Exception:
        return False

def _yesno(prompt: str) -> bool:
    while True:
        a = input(f"{prompt} [y/n] ").strip().lower()
        if a in {"y","yes"}: return True
        if a in {"n","no"}:  return False
        print("Please type 'y' or 'n'.")

def _ask_label(what: str) -> str:
    while True:
        a = input(f"Label the {what} as REAL or FAKE? [r/f] ").strip().lower()
        if a in {"r","real"}: return "real"
        if a in {"f","fake"}: return "fake"
        print("Please type 'r' or 'f'.")

# per-modality, per-label counters live in: dataset_creation/.counters/
def _counter_file(root: Path, kind: str, label: str) -> Path:
    # kind âˆˆ {"audio","frames"} ; label âˆˆ {"real","fake"}
    return _ensure_dir(root / ".counters") / f"{kind}_counter_{label}.txt"

def _read_counter(path: Path) -> int:
    if not path.exists():
        path.write_text("0", encoding="utf-8")
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return 0

def _write_counter(path: Path, value: int):
    path.write_text(str(value), encoding="utf-8")

def _gather_images(folder: Path, limit: int = 10) -> List[Path]:
    exts = {".jpg",".jpeg",".png",".bmp",".webp",".tiff"}
    files = [p for p in sorted(folder.glob("*")) if p.is_file() and p.suffix.lower() in exts]
    if len(files) > limit:
        random.shuffle(files)
        files = files[:limit]
    return files

def _pct(p: float) -> str:
    return f"{p*100:.1f}%"

def _reencode_video_mp4_std(src: Path, dst: Path, cfg: FBConfig) -> Path:
    _ensure_dir(dst.parent)
    vf = f"scale=-2:{cfg.video_height},fps={cfg.video_fps}"
    cmd = (
        f'ffmpeg -y -i {shlex.quote(str(src))} '
        f'-map 0:v:0 -c:v {cfg.video_codec} -pix_fmt yuv420p -vf "{vf}" '
        f'-map a:0? -c:a {cfg.audio_codec} -ac {cfg.audio_channels} -ar {cfg.audio_sr} '
        f'{shlex.quote(str(dst))}'
    )
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError("Video normalisation failed.")
    return dst

# ------------- API -------------
def run_feedback_interactive(
    video_path: str,
    *,
    cfg: FBConfig,
    precomputed: Optional[Tuple[Optional[Path], Optional[Path], int, Optional[Path]]] = None,
) -> None:
    """
    Preview â†’ confirm â†’ save with structure:
      dataset_creation/
        audio/
          real/ a1.mp3, a2.mp3, ...
          fake/ a1.mp3, a2.mp3, ...
        frames/
          real/ f1.jpg, f2.jpg, ...
          fake/ f1.jpg, f2.jpg, ...
    """
    vpath = Path(video_path)
    if not vpath.exists():
        print(f"[FEEDBACK] Video not found: {vpath}"); return
    if not _ffprobe_ok(vpath):
        print("[FEEDBACK] Input is damaged/unreadable. Skipping."); return

    # 1) Normalize video (we use it only to robustly extract audio/frames)
    norm_root = Path("outputs") / "normalized_video"
    norm_dst  = norm_root / f"{vpath.stem}.mp4"
    norm_mp4  = vpath
    if cfg.normalize_video:
        try:
            norm_mp4 = _reencode_video_mp4_std(vpath, norm_dst, cfg)
            print(f"[FEEDBACK] Normalised video -> {norm_mp4}")
        except Exception as e:
            print(f"[FEEDBACK] Normalisation failed ({e}). Skipping."); return

    # 2) Ensure artifacts (reuse precomputed if provided)
    audio_mp3: Optional[Path] = None
    frames_dir: Optional[Path] = None
    n_frames: int = 0
    audio_clips_dir: Optional[Path] = None

    if precomputed:
        audio_mp3, frames_dir, n_frames, audio_clips_dir = precomputed
    else:
        temp_dir = Path(".tmp_feedback")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        audio_mp3, frames_dir, n_frames = split_video_to_mp3_and_frames(
            str(norm_mp4),
            out_dir=str(temp_dir),
            img_fps=cfg.img_fps,
            img_ext=cfg.img_ext,
            quality=cfg.jpg_quality,
            start=None, end=None,
            bitrate=cfg.audio_bitrate
        )

    # 3) Show predictions (your requested format)
    print("\n=== Model predictions (for your review) ===")
    a_model, a_meta = load_audio_model()
    i_model, i_meta = load_image_model()

    # AUDIO
    if audio_mp3 is not None and audio_mp3.exists():
        ra = predict_audio(a_model, str(audio_mp3), a_meta, hop_seconds=2.0, topk=3)
        print(f"AUDIO â†’ {ra['label']} ({_pct(ra['prob'])})")
        for k, v in sorted(ra["dist"].items(), key=lambda kv: kv[1], reverse=True):
            print(f"   - {k}: {_pct(v)}")
    elif audio_clips_dir is not None and audio_clips_dir.exists():
        # average across clips
        dists = []
        for p in sorted(audio_clips_dir.glob("*.mp3")):
            r = predict_audio(a_model, str(p), a_meta, hop_seconds=2.0, topk=3)
            dists.append(r["dist"])
        if dists:
            keys = dists[0].keys()
            mean_dist = {k: sum(d[k] for d in dists)/len(dists) for k in keys}
            pred_k = max(mean_dist, key=lambda k: mean_dist[k])
            print(f"AUDIO (clips avg) â†’ {pred_k} ({_pct(mean_dist[pred_k])})")
            for k in sorted(mean_dist, key=lambda k: mean_dist[k], reverse=True):
                print(f"   - {k}: {_pct(mean_dist[k])}")
    else:
        print("AUDIO â†’ (none)")

    # FRAMES
    if frames_dir is not None and frames_dir.exists():
        imgs = _gather_images(frames_dir, limit=10)
        if imgs:
            print("\nFRAME predictions (sample):")
            dist_sum = None; n = 0
            for p in imgs:
                ri = predict_image(i_model, str(p), i_meta, topk=3)
                print(f"  {p.name} â†’ {ri['label']} ({_pct(ri['prob'])})")
                d = ri["dist"]
                if dist_sum is None: dist_sum = {k: d[k] for k in d}
                else:
                    for k in d: dist_sum[k] += d[k]
                n += 1
            if dist_sum and n:
                mean_dist = {k: dist_sum[k]/n for k in dist_sum}
                pred_k = max(mean_dist, key=lambda k: mean_dist[k])
                print(f"FRAMES (mean) â†’ {pred_k} ({_pct(mean_dist[pred_k])})")
        else:
            print("FRAMES â†’ (no images found)")
    else:
        print("FRAMES â†’ (none)")

    # 4) Ask if we should add to dataset at all
    print("\n=== Add this sample to the dataset? ===")
    if not _yesno("Add to dataset"):
        print("[FEEDBACK] Discarded by user."); return

    # 5) Ask labels for audio & frames (independent)
    audio_label = None
    frames_label = None
    if (audio_mp3 is not None) or (audio_clips_dir is not None and audio_clips_dir.exists()):
        audio_label = _ask_label("AUDIO")    # â†’ 'real' or 'fake'
    if (frames_dir is not None and frames_dir.exists()):
        frames_label = _ask_label("FRAMES")  # â†’ 'real' or 'fake'

    # 6) Destinations (no split)
    # dataset_creation/{audio|frames}/{real|fake}/...
    audio_root  = _ensure_dir(cfg.dataset_root / "audio")
    frames_root = _ensure_dir(cfg.dataset_root / "frames")
    dst_audio_dir  = _ensure_dir(audio_root  / (audio_label  or "real"))
    dst_frames_dir = _ensure_dir(frames_root / (frames_label or "real"))

    # per-label counters
    a_counter_path = _counter_file(cfg.dataset_root, "audio",  (audio_label  or "real"))
    f_counter_path = _counter_file(cfg.dataset_root, "frames", (frames_label or "real"))
    a_next = _read_counter(a_counter_path)
    f_next = _read_counter(f_counter_path)

    # 7) Save AUDIO as aN.mp3
    saved_audio = []
    if audio_label:
        if audio_mp3 is not None and audio_mp3.exists():
            a_next += 1
            dst_a = dst_audio_dir / f"a{a_next}.mp3"
            shutil.copy2(str(audio_mp3), str(dst_a))
            saved_audio.append(dst_a)
        elif audio_clips_dir is not None and audio_clips_dir.exists():
            for p in sorted(audio_clips_dir.glob("*.mp3")):
                a_next += 1
                dst_a = dst_audio_dir / f"a{a_next}.mp3"
                shutil.copy2(str(p), str(dst_a))
                saved_audio.append(dst_a)
        _write_counter(a_counter_path, a_next)

    # 8) Save FRAMES as fN.jpg
    saved_frames = []
    if frames_label and frames_dir is not None and frames_dir.exists():
        all_frames = [p for p in frames_dir.glob("*") if p.is_file()]
        if len(all_frames) > 10:
            random.shuffle(all_frames)
            all_frames = all_frames[:10]  # pick 10 random frames

        for p in sorted(all_frames, key=lambda x: x.name):
            f_next += 1
            dst_f = dst_frames_dir / f"f{f_next}.jpg"
            shutil.copy2(str(p), str(dst_f))
            saved_frames.append(dst_f)

        _write_counter(f_counter_path, f_next)

    # 9) Report + log
    print("\n[FEEDBACK] Saved:")
    if saved_audio:  print(f"  AUDIO   -> {dst_audio_dir} (+{len(saved_audio)} files)")
    if saved_frames: print(f"  FRAMES  -> {dst_frames_dir} (+{len(saved_frames)} files)")

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "orig_video": str(vpath.resolve()),
        "audio_label": audio_label or "",
        "frames_label": frames_label or "",
        "num_audio": len(saved_audio),
        "num_frames": len(saved_frames),
        "audio_saved_in": str(dst_audio_dir.resolve()) if saved_audio else "",
        "frames_saved_in": str(dst_frames_dir.resolve()) if saved_frames else "",
    }
    log_csv = _ensure_dir(cfg.dataset_root) / cfg.log_csv
    import csv as _csv
    file_exists = log_csv.exists()
    with log_csv.open("a", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(log_row.keys()))
        if not file_exists: w.writeheader()
        w.writerow(log_row)
    print(f"[FEEDBACK] Logged to: {log_csv.resolve()}")

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
