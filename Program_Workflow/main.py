from pathlib import Path
import random
import time
import torch
import cv2  # NEW: needed to write cropped images

from models_loader import (
    load_audio_model, load_image_model,
    predict_audio, predict_image
)
from video_io import split_video_to_mp3_and_frames

# NEW: feedback import
from feedback import FBConfig, run_feedback_interactive

# NEW: import face detector + cropper from your MediaPipe module
# (Facedetection.py must be in the same folder or on PYTHONPATH)
from Facedetection import detect_largest_face_bgr, crop_to_224

# ------------------------- settings -------------------------
VIDEO_PATH = ("test-1.mp4")
OUT_DIR = "outputs"
IMG_FPS = 2.0                # frames per second to extract
IMG_EXT = "jpg"
JPEG_QUALITY = 90
AUDIO_BITRATE = "192k"

MAX_FRAMES = 100             # sample up to 100 frames
RANDOM_SEED = 42             # set None for non-deterministic
TOPK = 3

IMAGE_RESULTS_TXT = "image_result.txt"                   # aggregate summary + timings
IMAGE_AGGREGATE_LIST_TXT = "image_aggregate_result.txt"  # per-frame list (up to 100)
AUDIO_RESULTS_TXT = "audio_result.txt"

# --- NEW: feedback wiring ---
FEEDBACK_ENABLED = True
DATASET_ROOT = "dataset_creation"     # per your requirement
FEEDBACK_SPLIT = None                 # or 'train'/'val'/'test' to force a split
# ------------------------------------------------------------

def list_frame_files(frames_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    return sorted([p for p in frames_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def aggregate_image_probs_top1_mean(model, meta, frame_paths, topk=3, device_str="cpu"):
    """
    Predict each frame (top-1) and compute mean top-1 confidence.
    Also returns total wall time for predicting all sampled frames, and avg per frame.
    """
    per_frame = []
    sum_top1 = 0.0
    n = 0

    # Accurate GPU timing: sync before & after the loop
    if device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for p in frame_paths:
        res = predict_image(model, str(p), meta, topk=topk)
        per_frame.append((str(p), res["label"], res["prob"], res["topk"], res["dist"]))
        sum_top1 += float(res["prob"])
        n += 1

    if device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    avg_time = (total_time / n) if n else 0.0

    mean_top1_conf = (sum_top1 / n) if n else 0.0
    return dict(
        per_frame=per_frame,
        mean_top1_conf=mean_top1_conf,
        total_time_s=total_time,
        avg_time_s=avg_time,
        n=n
    )

# --- NEW: helper to crop a single image path into 224x224 using MediaPipe ---
def crop_face_file_to_224(src_path: Path, dst_path: Path, margin: float = 0.1) -> bool:
    """
    Returns True if a face was detected & cropped to 224x224 and saved to dst_path; else False.
    """
    img = cv2.imread(str(src_path))
    if img is None:
        return False
    bbox = detect_largest_face_bgr(img, min_conf=0.5, model_selection=1)
    if bbox is None:
        return False
    face_224 = crop_to_224(img, bbox, margin=margin)
    if face_224 is None:
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), face_224)
    return True

def main():
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load models
    a_model, a_meta = load_audio_model(device=device)
    i_model, i_meta = load_image_model(device=device)

    # split video -> mp3 + frames
    audio_mp3, frames_dir, n_frames = split_video_to_mp3_and_frames(
        VIDEO_PATH,
        out_dir=OUT_DIR,
        img_fps=IMG_FPS,
        img_ext=IMG_EXT,
        quality=JPEG_QUALITY,
        start=None, end=None,
        bitrate=AUDIO_BITRATE
    )

    # -------------------- IMAGE PIPELINE (with face-gate) --------------------
    cropped_dir_for_video = None
    if frames_dir is None:
        print("[IMAGE] No frames directory returned.")
    else:
        all_frames = list_frame_files(frames_dir)
        if not all_frames:
            print(f"[IMAGE] No frame files found in: {frames_dir}")
        else:
            if RANDOM_SEED is not None:
                random.seed(RANDOM_SEED)
            k = min(MAX_FRAMES, len(all_frames))
            sampled = random.sample(all_frames, k) if len(all_frames) > k else all_frames
            sampled_sorted = sorted(sampled, key=lambda p: p.name)

            # NEW: try to crop faces from the sampled frames first
            cropped_root = Path(OUT_DIR) / "frames_cropped" / Path(VIDEO_PATH).stem
            cropped_paths = []
            any_face = False
            for src in sampled_sorted:
                dst = cropped_root / src.name  # keep same filenames
                ok = crop_face_file_to_224(src, dst, margin=0.12)
                if ok:
                    any_face = True
                    cropped_paths.append(dst)

            if not any_face:
                print("[IMAGE] No faces detected in sampled frames -> skipping image detection entirely.")
            else:
                print(f"[IMAGE] Using {len(cropped_paths)} cropped frames (of {len(sampled_sorted)} sampled).")
                cropped_dir_for_video = cropped_root

                agg = aggregate_image_probs_top1_mean(
                    i_model, i_meta, cropped_paths, topk=TOPK, device_str=device
                )

                # console summary
                print(f"[IMAGE] Top-1 mean confidence across cropped frames: {agg['mean_top1_conf']:.4f}")
                print(f"[IMAGE] Total prediction time (cropped frames): {agg['total_time_s']:.4f}s")
                print(f"[IMAGE] Avg prediction time per frame: {agg['avg_time_s']:.6f}s")

                # ----- 1) WRITE AGGREGATE SUMMARY + TIMINGS to image_result.txt -----
                summary_path = Path(IMAGE_RESULTS_TXT)
                summary_lines = []
                summary_lines.append("# IMAGE AGGREGATE RESULT (CROPPED)")
                summary_lines.append(f"# Source video: {VIDEO_PATH}")
                summary_lines.append(f"# Cropped frames dir:   {cropped_root}")
                summary_lines.append(f"# Total frames extracted: {len(all_frames)}")
                summary_lines.append(f"# Sampled frames attempted: {len(sampled_sorted)}")
                summary_lines.append(f"# Cropped frames used:  {agg['n']}")
                summary_lines.append("")
                summary_lines.append("## Aggregate (Top-1 mean confidence)")
                summary_lines.append(f"Mean top-1 confidence across cropped frames: {agg['mean_top1_conf']:.6f}")
                summary_lines.append("")
                summary_lines.append("## Timing")
                summary_lines.append(f"Total prediction time (cropped frames): {agg['total_time_s']:.6f} seconds")
                summary_lines.append(f"Average prediction time per frame:      {agg['avg_time_s']:.6f} seconds")
                summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
                print(f"[IMAGE] Wrote aggregate summary to: {summary_path.resolve()}")

                # ----- 2) WRITE PER-FRAME LIST to image_aggregate_result.txt -----
                perframe_path = Path(IMAGE_AGGREGATE_LIST_TXT)
                list_lines = []
                list_lines.append("# IMAGE PER-FRAME PREDICTIONS (CROPPED)")
                list_lines.append(f"# Source video: {VIDEO_PATH}")
                list_lines.append(f"# Cropped frames dir:   {cropped_root}")
                list_lines.append(f"# Total frames extracted: {len(all_frames)}")
                list_lines.append(f"# Cropped frames listed:  {agg['n']}")
                list_lines.append("")
                list_lines.append("## Per-frame predictions (path, label, confidence, top-k)")
                for path, label, prob, topk_list, _dist in agg["per_frame"]:
                    top_str = ", ".join([f"{k}={v:.3f}" for k, v in topk_list])
                    list_lines.append(f"{path}")
                    list_lines.append(f"  Prediction: {label}")
                    list_lines.append(f"  Confidence (prob): {prob:.6f}")
                    list_lines.append(f"  Top-{len(topk_list)}: {top_str}")
                    list_lines.append("")
                perframe_path.write_text("\n".join(list_lines), encoding="utf-8")
                print(f"[IMAGE] Wrote per-frame list to: {perframe_path.resolve()}")

    # -------------------- AUDIO PIPELINE (unchanged) --------------------
    if audio_mp3 is None:
        print("[AUDIO] No audio extracted.")
    else:
        # Accurate GPU timing around a single predict call
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res_aud = predict_audio(a_model, str(audio_mp3), a_meta, hop_seconds=2.0, topk=TOPK)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        audio_time = time.perf_counter() - t0

        print(f"[AUDIO] {Path(audio_mp3).name} â†’ {res_aud['label']} (prob={res_aud['prob']:.4f})")
        print(f"[AUDIO] Prediction time: {audio_time:.4f}s")

        out_aud_txt = Path(AUDIO_RESULTS_TXT)
        lines_a = []
        lines_a.append("# AUDIO RESULT")
        lines_a.append(f"Source video: {VIDEO_PATH}")
        lines_a.append(f"Audio file:   {audio_mp3}")
        lines_a.append("")
        lines_a.append(f"Prediction: {res_aud['label']}")
        lines_a.append(f"Confidence (prob): {res_aud['prob']:.6f}")
        lines_a.append("")
        lines_a.append("## Timing")
        lines_a.append(f"Prediction time: {audio_time:.6f} seconds")
        lines_a.append("")
        lines_a.append("Probability distribution:")
        for k, v in sorted(res_aud["dist"].items(), key=lambda kv: kv[1], reverse=True):
            lines_a.append(f"  - {k}: {v:.6f}")
        out_aud_txt.write_text("\n".join(lines_a), encoding="utf-8")
        print(f"[AUDIO] Wrote result to: {out_aud_txt.resolve()}")

    # -------------------- FEEDBACK (interactive dataset creation) --------------------
    if FEEDBACK_ENABLED:
        fb_cfg = FBConfig(
            dataset_root=Path(DATASET_ROOT),
            img_fps=IMG_FPS,
            img_ext=IMG_EXT,
            jpg_quality=JPEG_QUALITY,
            audio_bitrate=AUDIO_BITRATE,
            normalize_video=True,
            video_height=720,
            video_fps=30,
            seed=RANDOM_SEED if RANDOM_SEED is not None else 42,
        )

        # Reuse artifacts extracted above; IMPORTANT:
        # If we produced cropped frames, point feedback to the CROPPED folder,
        # so the preview & labeling use the same inputs we used for detection.
        frames_for_feedback = Path(cropped_dir_for_video) if cropped_dir_for_video else frames_dir
        precomputed = (audio_mp3, frames_for_feedback, n_frames, None)

        try:
            run_feedback_interactive(
                video_path=VIDEO_PATH,
                cfg=fb_cfg,
                precomputed=precomputed
            )
        except Exception as e:
            print(f"[FEEDBACK] Error: {e}")

if __name__ == "__main__":
    main()
