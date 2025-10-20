# video_io.py
"""
Utilities to extract audio (always MP3) and frames from a video.

Dependencies:
    pip install moviepy pillow tqdm imageio[ffmpeg]

Example:
    from video_io import split_video_to_mp3_and_frames

    audio_mp3, frames_dir, n_frames = split_video_to_mp3_and_frames(
        video_path="video.mp4",
        out_dir="outputs",
        img_fps=2.0,      # frames per second to save (<=0 to skip)
        img_ext="jpg",    # jpg|png|webp|bmp|tiff
        quality=90,       # JPEG quality if jpg/jpeg
        start=None, end=None,
        bitrate="192k"    # MP3 bitrate
    )
"""

from __future__ import annotations
from pathlib import Path
from datetime import timedelta
from typing import Optional, Tuple
import subprocess
import shlex

from PIL import Image, ImageFile

# MoviePy v2 changed imports; prefer new path, fall back for v1
# MoviePy v2 changed imports; prefer new path, fall back for v1
try:
    from moviepy import VideoFileClip  # MoviePy >= 2.0
except ImportError:
    from moviepy.editor import VideoFileClip  # MoviePy < 2.0


from tqdm import tqdm

# Locate ffmpeg reliably (imageio-ffmpeg), else fall back to PATH
try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    _FFPROBE = imageio_ffmpeg.get_ffprobe_exe()
except Exception:
    _FFMPEG = "ffmpeg"
    _FFPROBE = "ffprobe"

# Be resilient to partially written images
ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ["split_video_to_mp3_and_frames"]


# --------------------------- internal helpers ---------------------------

def _hhmmss(seconds: float) -> str:
    return str(timedelta(seconds=round(float(seconds))))

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _extract_audio_mp3(
    clip: VideoFileClip,
    out_audio_dir: Path,
    base_name: str,
    bitrate: str = "192k",
    src_video_path: Optional[Path] = None,
) -> Path:
    """
    Save audio as MP3. Strategy:
      1) Try MoviePy (fast path).
      2) If MoviePy reports no audio or fails, fall back to ffmpeg.
         - Uses first audio stream if present (-map a:0?).
         - Mono (1 ch) @ 16 kHz for ML-friendly output.
    """
    _ensure_dir(out_audio_dir)
    out_path = out_audio_dir / f"{base_name}.mp3"

    # 1) MoviePy path
    try:
        if clip.audio is not None:
            clip.audio.write_audiofile(
                str(out_path),
                codec="mp3",
                bitrate=bitrate,
                # You could set fps=16000 here, but the model's loader
                # resamples anyway; leaving default often preserves quality.
                verbose=False,
                logger=None,
            )
            if out_path.exists() and out_path.stat().st_size > 0:
                return out_path
    except Exception as e:
        print(f"[AUDIO] MoviePy write failed ({e}). Trying ffmpeg fallbackÃ¢â‚¬Â¦")

    # 2) FFmpeg fallback (requires source file path)
    if src_video_path and src_video_path.exists():
        # -map a:0? selects first audio stream if available (no hard error if missing)
        # -ac 1, -ar 16000 make mono/16k which is common for audio ML.
        cmd = (
            f'{shlex.quote(_FFMPEG)} -y -i {shlex.quote(str(src_video_path))} '
            f'-vn -map a:0? -ac 1 -ar 16000 -b:a {shlex.quote(bitrate)} '
            f'{shlex.quote(str(out_path))}'
        )
        print(f"[AUDIO] FFmpeg fallback: {cmd}")
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        else:
            tail = proc.stderr.decode(errors="ignore").splitlines()[-10:]
            print("[AUDIO] ffmpeg stderr (tail):\n" + "\n".join(tail))

    # If we reach here, thereÃ¢â‚¬â„¢s no extractable audio
    raise RuntimeError("No audio stream found or extraction failed.")

def _extract_frames(
    clip: VideoFileClip,
    out_frames_root: Path,
    base_name: str,
    img_fps: float = 1.0,
    img_ext: str = "jpg",
    quality: int = 95,
) -> Tuple[Path, int]:
    """
    Save frames at a fixed FPS under: <out_frames_root>/<base_name>/*.ext
    Returns: (frames_dir, count)
    """
    frames_dir = _ensure_dir(out_frames_root / base_name)

    duration = float(clip.duration or 0.0)
    total = int(duration * img_fps) if duration > 0 and img_fps > 0 else None
    pad = len(str(max(total, 1))) if total else 6

    count = 0
    for frame in tqdm(
        clip.iter_frames(fps=img_fps, dtype="uint8"),
        total=total,
        desc=f"Frames @ {img_fps} FPS",
    ):
        img = Image.fromarray(frame)  # frame is (H, W, 3) RGB uint8
        fname = f"{base_name}_{str(count).zfill(pad)}.{img_ext.lower()}"
        fpath = frames_dir / fname

        if img_ext.lower() in {"jpg", "jpeg"}:
            img.save(fpath, format="JPEG", quality=quality, optimize=True)
        else:
            img.save(fpath)

        count += 1

    return frames_dir, count


# ------------------------------- public API -----------------------------------

def split_video_to_mp3_and_frames(
    video_path: str,
    out_dir: str = "outputs",
    *,
    img_fps: float = 1.0,
    img_ext: str = "jpg",
    quality: int = 95,
    start: Optional[float] = None,
    end: Optional[float] = None,
    bitrate: str = "192k",
) -> Tuple[Optional[Path], Optional[Path], int]:
    """
    Extract audio (MP3) and frames (images) from a video.

    Args:
        video_path: input video path (.mp4, .mov, etc.)
        out_dir: root output directory (creates 'audio' and 'frames' subfolders)
        img_fps: frames per second to save (<=0 to skip frames extraction)
        img_ext: image extension: 'jpg'|'png'|'webp'|'bmp'|'tiff'
        quality: JPEG quality (1-95) if img_ext is 'jpg'/'jpeg'
        start: optional start time (seconds) to subclip
        end: optional end time (seconds) to subclip
        bitrate: MP3 bitrate (e.g., '128k', '192k', '256k')

    Returns:
        (audio_mp3_path, frames_dir, n_frames)
        - audio_mp3_path: Path to saved MP3 (or None if no audio)
        - frames_dir: directory where frames were saved (or None if skipped)
        - n_frames: number of frames written
    """
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video not found: {vpath}")

    out_root = Path(out_dir)
    out_audio_dir = out_root / "audio"
    out_frames_dir = out_root / "frames"
    base_name = vpath.stem

    # Load clip
    clip = VideoFileClip(str(vpath))
    full_dur = float(clip.duration or 0.0)

    # Optional subclip
    if start is not None or end is not None:
        s = max(0.0, float(start or 0.0))
        e = float(end) if end is not None else full_dur
        e = min(e, full_dur)
        if e <= s:
            clip.close()
            raise ValueError("Invalid subclip: end time must be greater than start time.")
        clip = clip.subclip(s, e)
        print(f"Subclip: {_hhmmss(s)} Ã¢â€ â€™ {_hhmmss(e)} (len {_hhmmss(e - s)})")
    else:
        print(f"Full clip length: {_hhmmss(full_dur)}")

    # Audio (MP3) with FFmpeg fallback
    audio_path: Optional[Path]
    try:
        audio_path = _extract_audio_mp3(
            clip,
            out_audio_dir,
            base_name,
            bitrate=bitrate,
            src_video_path=vpath,
        )
        print(f"[AUDIO] Saved Ã¢â€ â€™ {audio_path}")
    except Exception as ex:
        print(f"[AUDIO] Skipped: {ex}")
        audio_path = None

    # Frames
    frames_dir: Optional[Path] = None
    n_frames = 0
    if img_fps > 0:
        frames_dir, n_frames = _extract_frames(
            clip, out_frames_dir, base_name, img_fps=img_fps, img_ext=img_ext, quality=quality
        )
        print(f"[FRAMES] Saved Ã¢â€ â€™ {n_frames} files in {frames_dir}")
    else:
        print("[FRAMES] img_fps <= 0 Ã¢â€ â€™ skipping frame extraction.")

    clip.close()
    return audio_path, frames_dir, n_frames
