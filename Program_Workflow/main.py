import argparse
from pathlib import Path
import random
import time
import torch
import cv2

from models_loader import (
    load_audio_model, load_image_model,
    predict_audio, predict_image
)
from video_io import split_video_to_mp3_and_frames
from feedback import FBConfig, run_feedback_interactive
from Facedetection import FaceDetector, crop_to_224


# ------------------------- CLI settings -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Deepfake detection: audio + image pipeline with feedback")
    p.add_argument("video", nargs="?", default="test-1.mp4",
                   help="Path to input video (default: test-1.mp4)")
    p.add_argument("--out-dir", default="outputs",
                   help="Root output directory (default: outputs)")
    p.add_argument("--fps", type=float, default=2.0,
                   help="Frames per second to extract (default: 2.0)")
    p.add_argument("--max-frames", type=int, default=100,
                   help="Max frames to sample for image prediction (default: 100)")
    p.add_argument("--topk", type=int, default=3,
                   help="Top-k classes in output (default: 3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed, set to -1 for non-deterministic (default: 42)")
    p.add_argument("--no-feedback", action="store_true",
                   help="Skip interactive feedback / dataset-building step")
    p.add_argument("--dataset-root", default="dataset_creation",
                   help="Root folder for feedback dataset (default: dataset_creation)")
    return p.parse_args()


# ------------------------- helpers -------------------------

def list_frame_files(frames_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    return sorted([p for p in frames_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def aggregate_image_probs_top1_mean(model, meta, frame_paths, topk=3, device_str="cpu"):
    """Predict each frame (top-1) and return mean top-1 confidence + timing."""
    per_frame = []
    sum_top1 = 0.0
    n = 0

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
        n=n,
    )


def crop_face_file_to_224(detector: FaceDetector, src_path: Path, dst_path: Path,
                           margin: float = 0.1) -> bool:
    """
    Detect and crop the largest face from src_path, save 224×224 to dst_path.
    Returns True on success. Uses a shared FaceDetector (no per-call MediaPipe init).
    """
    img = cv2.imread(str(src_path))
    if img is None:
        return False
    bbox = detector.detect_largest_face(img)
    if bbox is None:
        return False
    face_224 = crop_to_224(img, bbox, margin=margin)
    if face_224 is None:
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), face_224)
    return True


# ------------------------- main -------------------------

def main():
    args = parse_args()

    VIDEO_PATH = args.video
    OUT_DIR = args.out_dir
    IMG_FPS = args.fps
    IMG_EXT = "jpg"
    JPEG_QUALITY = 90
    AUDIO_BITRATE = "192k"
    MAX_FRAMES = args.max_frames
    RANDOM_SEED = None if args.seed == -1 else args.seed
    TOPK = args.topk
    FEEDBACK_ENABLED = not args.no_feedback
    DATASET_ROOT = args.dataset_root

    IMAGE_RESULTS_TXT = "image_result.txt"
    IMAGE_AGGREGATE_LIST_TXT = "image_aggregate_result.txt"
    AUDIO_RESULTS_TXT = "audio_result.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models once — reused by both detection and feedback
    a_model, a_meta = load_audio_model(device=device)
    i_model, i_meta = load_image_model(device=device)

    # Split video -> mp3 + frames
    audio_mp3, frames_dir, n_frames = split_video_to_mp3_and_frames(
        VIDEO_PATH,
        out_dir=OUT_DIR,
        img_fps=IMG_FPS,
        img_ext=IMG_EXT,
        quality=JPEG_QUALITY,
        start=None, end=None,
        bitrate=AUDIO_BITRATE,
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

            cropped_root = Path(OUT_DIR) / "frames_cropped" / Path(VIDEO_PATH).stem
            cropped_paths = []
            any_face = False

            # Create detector once for all frames (fixes per-frame MediaPipe init)
            with FaceDetector(min_conf=0.5, model_selection=1) as detector:
                for src in sampled_sorted:
                    dst = cropped_root / src.name
                    ok = crop_face_file_to_224(detector, src, dst, margin=0.12)
                    if ok:
                        any_face = True
                        cropped_paths.append(dst)

            if not any_face:
                print("[IMAGE] No faces detected in sampled frames -> skipping image detection.")
            else:
                print(f"[IMAGE] Using {len(cropped_paths)} cropped frames "
                      f"(of {len(sampled_sorted)} sampled).")
                cropped_dir_for_video = cropped_root

                agg = aggregate_image_probs_top1_mean(
                    i_model, i_meta, cropped_paths, topk=TOPK, device_str=device
                )

                print(f"[IMAGE] Top-1 mean confidence: {agg['mean_top1_conf']:.4f}")
                print(f"[IMAGE] Total prediction time: {agg['total_time_s']:.4f}s")
                print(f"[IMAGE] Avg per frame: {agg['avg_time_s']:.6f}s")

                # Write aggregate summary
                summary_lines = [
                    "# IMAGE AGGREGATE RESULT (CROPPED)",
                    f"# Source video: {VIDEO_PATH}",
                    f"# Cropped frames dir:   {cropped_root}",
                    f"# Total frames extracted: {len(all_frames)}",
                    f"# Sampled frames attempted: {len(sampled_sorted)}",
                    f"# Cropped frames used:  {agg['n']}",
                    "",
                    "## Aggregate (Top-1 mean confidence)",
                    f"Mean top-1 confidence across cropped frames: {agg['mean_top1_conf']:.6f}",
                    "",
                    "## Timing",
                    f"Total prediction time (cropped frames): {agg['total_time_s']:.6f} seconds",
                    f"Average prediction time per frame:      {agg['avg_time_s']:.6f} seconds",
                ]
                Path(IMAGE_RESULTS_TXT).write_text("\n".join(summary_lines), encoding="utf-8")
                print(f"[IMAGE] Wrote aggregate summary to: {Path(IMAGE_RESULTS_TXT).resolve()}")

                # Write per-frame list
                list_lines = [
                    "# IMAGE PER-FRAME PREDICTIONS (CROPPED)",
                    f"# Source video: {VIDEO_PATH}",
                    f"# Cropped frames dir:   {cropped_root}",
                    f"# Total frames extracted: {len(all_frames)}",
                    f"# Cropped frames listed:  {agg['n']}",
                    "",
                    "## Per-frame predictions (path, label, confidence, top-k)",
                ]
                for path, label, prob, topk_list, _dist in agg["per_frame"]:
                    top_str = ", ".join([f"{k}={v:.3f}" for k, v in topk_list])
                    list_lines += [
                        f"{path}",
                        f"  Prediction: {label}",
                        f"  Confidence (prob): {prob:.6f}",
                        f"  Top-{len(topk_list)}: {top_str}",
                        "",
                    ]
                Path(IMAGE_AGGREGATE_LIST_TXT).write_text("\n".join(list_lines), encoding="utf-8")
                print(f"[IMAGE] Wrote per-frame list to: {Path(IMAGE_AGGREGATE_LIST_TXT).resolve()}")

    # -------------------- AUDIO PIPELINE --------------------
    if audio_mp3 is None:
        print("[AUDIO] No audio extracted.")
    else:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res_aud = predict_audio(a_model, str(audio_mp3), a_meta, hop_seconds=2.0, topk=TOPK)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        audio_time = time.perf_counter() - t0

        print(f"[AUDIO] {Path(audio_mp3).name} -> {res_aud['label']} (prob={res_aud['prob']:.4f})")
        print(f"[AUDIO] Prediction time: {audio_time:.4f}s")

        lines_a = [
            "# AUDIO RESULT",
            f"Source video: {VIDEO_PATH}",
            f"Audio file:   {audio_mp3}",
            "",
            f"Prediction: {res_aud['label']}",
            f"Confidence (prob): {res_aud['prob']:.6f}",
            "",
            "## Timing",
            f"Prediction time: {audio_time:.6f} seconds",
            "",
            "Probability distribution:",
        ]
        for k, v in sorted(res_aud["dist"].items(), key=lambda kv: kv[1], reverse=True):
            lines_a.append(f"  - {k}: {v:.6f}")
        Path(AUDIO_RESULTS_TXT).write_text("\n".join(lines_a), encoding="utf-8")
        print(f"[AUDIO] Wrote result to: {Path(AUDIO_RESULTS_TXT).resolve()}")

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

        frames_for_feedback = Path(cropped_dir_for_video) if cropped_dir_for_video else frames_dir
        precomputed = (audio_mp3, frames_for_feedback, n_frames, None)

        try:
            run_feedback_interactive(
                video_path=VIDEO_PATH,
                cfg=fb_cfg,
                precomputed=precomputed,
                # Pass already-loaded models — avoids reloading inside feedback
                a_model=a_model,
                a_meta=a_meta,
                i_model=i_model,
                i_meta=i_meta,
            )
        except Exception as e:
            print(f"[FEEDBACK] Error: {e}")


if __name__ == "__main__":
    main()
