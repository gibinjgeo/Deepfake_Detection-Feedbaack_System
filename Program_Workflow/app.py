# app.py — Streamlit GUI for Deepfake Detection & Feedback System
"""
Run from the Program_Workflow directory:
    cd Program_Workflow
    streamlit run app.py

Or from the project root:
    streamlit run Program_Workflow/app.py
"""

from __future__ import annotations

import csv
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import torch
import streamlit as st

# ── ensure imports resolve from Program_Workflow regardless of cwd ────────────
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
os.chdir(HERE)   # model checkpoints use relative paths (best_audio_model.pt etc.)

from models_loader import load_audio_model, load_image_model, predict_audio, predict_image
from video_io import split_video_to_mp3_and_frames
from Facedetection import FaceDetector, crop_to_224
from feedback import (
    FBConfig,
    _ensure_dir,
    _counter_file,
    _read_counter,
    _write_counter,
    _reencode_video_mp4_std,
)
from retrain_module import check_status, maybe_auto_retrain

# ── constants ─────────────────────────────────────────────────────────────────
DATASET_ROOT = HERE / "dataset_creation"
OUTPUTS_DIR  = HERE / "outputs"
UPLOADS_DIR  = OUTPUTS_DIR / "uploads"
AUDIO_CKPT   = HERE / "best_audio_model.pt"
IMAGE_CKPT   = HERE / "best_model.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Deepfake Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Cached resources — loaded once per session
# =============================================================================

@st.cache_resource(show_spinner="Loading models onto GPU/CPU...")
def get_models():
    """Load audio + image models. Cached so they're only loaded once."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a_model, a_meta = load_audio_model(device=device)
    i_model, i_meta = load_image_model(device=device)
    return a_model, a_meta, i_model, i_meta, device


# =============================================================================
# Pipeline helpers
# =============================================================================

def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _label_badge(label: str) -> str:
    return f"🟢 {label.upper()}" if label.lower() == "real" else f"🔴 {label.upper()}"


def _list_frames(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def run_pipeline(video_path: str, fps: float, max_frames: int, seed: int) -> dict:
    """
    Full detection pipeline:
      1. Split video -> audio MP3 + raw frames
      2. Face detection + crop to 224x224
      3. Audio inference (sliding window)
      4. Image inference on each cropped face

    Returns a result dict with all artifacts and predictions.
    """
    a_model, a_meta, i_model, i_meta, device = get_models()
    random.seed(seed)

    # ── 1. Split video ────────────────────────────────────────────────────────
    audio_mp3, frames_dir, n_frames = split_video_to_mp3_and_frames(
        video_path,
        out_dir=str(OUTPUTS_DIR),
        img_fps=fps,
        img_ext="jpg",
        quality=90,
        bitrate="192k",
    )

    # ── 2. Face detection ─────────────────────────────────────────────────────
    cropped_paths: list[Path] = []
    cropped_root = OUTPUTS_DIR / "frames_cropped" / Path(video_path).stem

    if frames_dir and frames_dir.exists():
        all_frames = _list_frames(frames_dir)
        k       = min(max_frames, len(all_frames))
        sampled = sorted(
            random.sample(all_frames, k) if len(all_frames) > k else all_frames,
            key=lambda p: p.name,
        )

        with FaceDetector(min_conf=0.5, model_selection=1) as detector:
            for src in sampled:
                img = cv2.imread(str(src))
                if img is None:
                    continue
                bbox = detector.detect_largest_face(img)
                if bbox is None:
                    continue
                face = crop_to_224(img, bbox, margin=0.12)
                if face is None:
                    continue
                dst = cropped_root / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), face)
                cropped_paths.append(dst)

    # ── 3. Audio inference ────────────────────────────────────────────────────
    audio_result: Optional[dict] = None
    if audio_mp3 and audio_mp3.exists():
        audio_result = predict_audio(a_model, str(audio_mp3), a_meta, hop_seconds=2.0, topk=3)

    # ── 4. Image inference ────────────────────────────────────────────────────
    image_results: list[dict] = []
    for p in cropped_paths:
        res = predict_image(i_model, str(p), i_meta, topk=3)
        image_results.append({"path": str(p), **res})

    return {
        "audio_mp3":     audio_mp3,
        "frames_dir":    frames_dir,
        "n_frames":      n_frames,
        "cropped_paths": cropped_paths,
        "audio_result":  audio_result,
        "image_results": image_results,
    }


def _mean_image_dist(image_results: list[dict]) -> tuple[str, float, dict]:
    """Aggregate per-frame predictions into a mean distribution."""
    probs_by_label: dict[str, list[float]] = {}
    for r in image_results:
        for label, prob in r["dist"].items():
            probs_by_label.setdefault(label, []).append(prob)
    mean_dist = {k: sum(v) / len(v) for k, v in probs_by_label.items()}
    top_label = max(mean_dist, key=lambda k: mean_dist[k])
    return top_label, mean_dist[top_label], mean_dist


def save_to_dataset(
    audio_mp3:    Optional[Path],
    frames_dir:   Optional[Path],
    audio_label:  str,
    frames_label: str,
    dataset_root: Path,
) -> dict:
    """
    Copy audio + frames into dataset_creation/{audio|frames}/{label}/
    using counter-based sequential file naming (a1.mp3, f1.jpg, ...).
    Appends a row to feedback_log.csv.
    Returns {"audio": N_saved, "frames": N_saved}.
    """
    dst_audio_dir  = _ensure_dir(dataset_root / "audio"  / audio_label)
    dst_frames_dir = _ensure_dir(dataset_root / "frames" / frames_label)

    a_ctr = _counter_file(dataset_root, "audio",  audio_label)
    f_ctr = _counter_file(dataset_root, "frames", frames_label)
    a_next = _read_counter(a_ctr)
    f_next = _read_counter(f_ctr)

    # Save audio
    saved_audio = 0
    if audio_mp3 and audio_mp3.exists():
        a_next += 1
        shutil.copy2(str(audio_mp3), str(dst_audio_dir / f"a{a_next}.mp3"))
        saved_audio = 1
        _write_counter(a_ctr, a_next)

    # Save frames (up to 10 random)
    saved_frames = 0
    if frames_dir and frames_dir.exists():
        all_f = [p for p in frames_dir.glob("*") if p.is_file()]
        if len(all_f) > 10:
            random.shuffle(all_f)
            all_f = all_f[:10]
        for p in sorted(all_f, key=lambda x: x.name):
            f_next += 1
            shutil.copy2(str(p), str(dst_frames_dir / f"f{f_next}.jpg"))
            saved_frames += 1
        _write_counter(f_ctr, f_next)

    # Append to CSV log
    log_csv   = _ensure_dir(dataset_root) / "feedback_log.csv"
    log_row   = {
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "audio_label":    audio_label,
        "frames_label":   frames_label,
        "num_audio":      saved_audio,
        "num_frames":     saved_frames,
        "audio_saved_in": str(dst_audio_dir),
        "frames_saved_in": str(dst_frames_dir),
    }
    file_exists = log_csv.exists()
    with log_csv.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log_row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(log_row)

    return {"audio": saved_audio, "frames": saved_frames}


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("Deepfake Detector")
    st.caption("Audio + Image Analysis with Human Feedback Loop")
    st.divider()

    # ── Model status ──────────────────────────────────────────────────────────
    st.subheader("Model Status")
    device_label = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.write(f"Device: `{device_label}`")
    st.write(f"{'✅' if AUDIO_CKPT.exists() else '❌'} Audio checkpoint")
    st.write(f"{'✅' if IMAGE_CKPT.exists() else '❌'} Image checkpoint")

    if not AUDIO_CKPT.exists() or not IMAGE_CKPT.exists():
        st.warning("One or more model checkpoints are missing. Place `best_audio_model.pt` and `best_model.pt` in the `Program_Workflow/` folder.")

    st.divider()

    # ── Dataset status summary ────────────────────────────────────────────────
    st.subheader("Dataset Status")
    try:
        _status = check_status()
        for mod, info in _status.items():
            icon = "✅" if info["ready"] else "⏳"
            st.write(f"{icon} **{mod.upper()}**")
            for cls, cnt in info["counts"].items():
                st.caption(f"  {cls}: {cnt} / {info['threshold']}")
    except Exception as _e:
        st.caption(f"Status unavailable: {_e}")

    st.divider()
    st.caption("Built with Streamlit · PyTorch · MediaPipe")


# =============================================================================
# Main — tabs
# =============================================================================

tab_analyze, tab_feedback, tab_dataset = st.tabs([
    "🔍  Analyze Video",
    "📝  Feedback & Labeling",
    "📊  Dataset & Fine-Tune",
])


# =============================================================================
# TAB 1 — ANALYZE VIDEO
# =============================================================================

with tab_analyze:
    st.header("Analyze Video")
    st.caption("Upload a video to run deepfake detection on its audio and face frames.")

    # ── Upload + settings ─────────────────────────────────────────────────────
    col_up, col_cfg = st.columns([3, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Upload video file",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            help="Accepted: MP4, MOV, AVI, MKV, WebM",
        )

    with col_cfg:
        st.subheader("Settings")
        fps         = st.slider("Extraction FPS",  0.5, 5.0, 2.0, 0.5,
                                help="Frames per second to sample from the video")
        max_frames  = st.slider("Max face frames",  10, 200, 100, 10,
                                help="Cap on how many frames to run face detection on")
        seed        = st.number_input("Seed", value=42, min_value=0,
                                      help="Random seed for frame sampling")

    # ── Video preview + run ───────────────────────────────────────────────────
    if uploaded is not None:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        video_path = str(UPLOADS_DIR / uploaded.name)
        with open(video_path, "wb") as fh:
            fh.write(uploaded.read())

        st.video(video_path)

        models_ready = AUDIO_CKPT.exists() and IMAGE_CKPT.exists()
        if st.button("Run Analysis", type="primary", use_container_width=True,
                     disabled=not models_ready):
            with st.spinner("Running pipeline: splitting video → face detection → inference..."):
                try:
                    result = run_pipeline(video_path, fps, int(max_frames), int(seed))
                    st.session_state["pipeline_result"] = result
                    st.session_state["video_path"]      = video_path
                    st.session_state["feedback_done"]   = False
                    st.success("Analysis complete! Scroll down to see results.")
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    st.exception(exc)

        if not models_ready:
            st.warning("Model checkpoints not found — place them in `Program_Workflow/` first.")

    # ── Results ───────────────────────────────────────────────────────────────
    if "pipeline_result" in st.session_state:
        res = st.session_state["pipeline_result"]
        st.divider()
        st.subheader("Results")

        col_aud, col_img = st.columns(2)

        # ── Audio ──────────────────────────────────────────────────────────
        with col_aud:
            st.markdown("#### Audio Analysis")
            ar = res.get("audio_result")
            if ar:
                st.metric(
                    label="Prediction",
                    value=_label_badge(ar["label"]),
                    delta=f"{_pct(ar['prob'])} confidence",
                )
                st.caption("Class probabilities:")
                for lbl, prob in sorted(ar["dist"].items(), key=lambda x: x[1], reverse=True):
                    st.progress(float(prob), text=f"{lbl}: {_pct(prob)}")
            else:
                st.warning("No audio stream found in this video.")

        # ── Image ──────────────────────────────────────────────────────────
        with col_img:
            st.markdown("#### Face / Image Analysis")
            img_results = res.get("image_results", [])
            if img_results:
                top_label, top_prob, mean_dist = _mean_image_dist(img_results)
                st.metric(
                    label=f"Prediction (mean across {len(img_results)} frames)",
                    value=_label_badge(top_label),
                    delta=f"{_pct(top_prob)} confidence",
                )
                st.caption("Mean class probabilities across frames:")
                for lbl, prob in sorted(mean_dist.items(), key=lambda x: x[1], reverse=True):
                    st.progress(float(prob), text=f"{lbl}: {_pct(prob)}")
            else:
                st.warning("No faces detected in any sampled frame.")

        # ── Per-frame breakdown ────────────────────────────────────────────
        if res.get("image_results"):
            with st.expander("Per-frame predictions + face crops"):
                sample = res["image_results"][:9]   # show up to 9 thumbnails
                cols   = st.columns(3)
                for i, r in enumerate(sample):
                    with cols[i % 3]:
                        st.image(
                            r["path"],
                            caption=f"{_label_badge(r['label'])} {_pct(r['prob'])}",
                            width=160,
                        )
                if len(res["image_results"]) > 9:
                    st.caption(f"... and {len(res['image_results']) - 9} more frames.")

        st.info("Head to the **Feedback & Labeling** tab to add this video to your training dataset.")


# =============================================================================
# TAB 2 — FEEDBACK & LABELING
# =============================================================================

with tab_feedback:
    st.header("Feedback & Labeling")
    st.caption(
        "Review the model predictions and assign ground-truth labels. "
        "Your labels are saved to the feedback dataset and can trigger automatic fine-tuning."
    )

    if "pipeline_result" not in st.session_state:
        st.info("Run an analysis first on the **Analyze Video** tab.")

    elif st.session_state.get("feedback_done"):
        st.success("Feedback already submitted for this video.")
        if st.button("Reset — analyze another video"):
            del st.session_state["pipeline_result"]
            del st.session_state["feedback_done"]
            st.rerun()

    else:
        res         = st.session_state["pipeline_result"]
        ar          = res.get("audio_result")
        img_results = res.get("image_results", [])
        has_audio   = ar is not None
        has_faces   = len(img_results) > 0

        # ── Prediction summary ─────────────────────────────────────────────
        st.subheader("Model Predictions (for review)")
        col_pa, col_pi = st.columns(2)

        with col_pa:
            st.markdown("**Audio**")
            if has_audio:
                st.write(f"{_label_badge(ar['label'])} — {_pct(ar['prob'])}")
                for lbl, prob in sorted(ar["dist"].items(), key=lambda x: x[1], reverse=True):
                    st.caption(f"  {lbl}: {_pct(prob)}")
            else:
                st.write("No audio extracted.")

        with col_pi:
            st.markdown("**Frames (mean)**")
            if has_faces:
                top_label, top_prob, mean_dist = _mean_image_dist(img_results)
                st.write(f"{_label_badge(top_label)} — {_pct(top_prob)}")
                st.caption(f"  {len(img_results)} face frames analyzed")
                for lbl, prob in sorted(mean_dist.items(), key=lambda x: x[1], reverse=True):
                    st.caption(f"  {lbl}: {_pct(prob)}")
            else:
                st.write("No faces detected.")

        st.divider()

        # ── Label assignment ───────────────────────────────────────────────
        st.subheader("Assign Ground-Truth Labels")
        st.caption(
            "You can label audio and frames independently. "
            "The model may be wrong — your label is the ground truth."
        )

        col_la, col_lf = st.columns(2)
        with col_la:
            audio_label = st.radio(
                "Audio label",
                options=["real", "fake"],
                index=0,
                key="fb_audio_label",
                disabled=not has_audio,
                help="Disabled if no audio was found in the video.",
            )
        with col_lf:
            frames_label = st.radio(
                "Frames label",
                options=["real", "fake"],
                index=0,
                key="fb_frames_label",
                disabled=not has_faces,
                help="Disabled if no faces were detected.",
            )

        st.divider()

        # ── Action buttons ─────────────────────────────────────────────────
        col_save, col_discard = st.columns(2)

        with col_save:
            if st.button("Save to Dataset", type="primary", use_container_width=True):
                with st.spinner("Saving to dataset..."):
                    try:
                        summary = save_to_dataset(
                            audio_mp3=res.get("audio_mp3"),
                            frames_dir=res.get("frames_dir"),
                            audio_label=audio_label,
                            frames_label=frames_label,
                            dataset_root=DATASET_ROOT,
                        )
                        st.session_state["feedback_done"] = True
                        st.success(
                            f"Saved! {summary['audio']} audio file(s), "
                            f"{summary['frames']} frame(s) added to dataset."
                        )

                        # Check if thresholds are now met
                        new_status = check_status()
                        for mod, info in new_status.items():
                            if info["ready"]:
                                st.info(
                                    f"**{mod.upper()} threshold reached!** "
                                    f"Go to the Dataset & Fine-Tune tab to update the model."
                                )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Save failed: {exc}")
                        st.exception(exc)

        with col_discard:
            if st.button("Discard (do not save)", use_container_width=True):
                st.session_state["feedback_done"] = True
                st.info("Discarded — this video will not be added to the dataset.")
                st.rerun()


# =============================================================================
# TAB 3 — DATASET & FINE-TUNE
# =============================================================================

with tab_dataset:
    st.header("Dataset Status & Fine-Tuning")

    # ── Refresh button ─────────────────────────────────────────────────────
    if st.button("Refresh Status"):
        st.cache_resource.clear()   # force sidebar + status to reload
        st.rerun()

    # ── Per-modality status cards ──────────────────────────────────────────
    try:
        status = check_status()
    except Exception as exc:
        st.error(f"Could not read dataset: {exc}")
        status = {}

    col_s_audio, col_s_image = st.columns(2)

    def _render_status_card(container, modality: str, info: dict):
        with container:
            st.subheader(f"{modality.upper()} Dataset")
            ready = info.get("ready", False)
            thr   = info.get("threshold", 0)

            for cls, cnt in info.get("counts", {}).items():
                missing = info["missing"].get(cls, 0)
                delta   = f"need {missing} more" if missing else "Ready"
                st.metric(f"{cls.capitalize()}", cnt, delta=delta)
                st.progress(min(cnt / max(thr, 1), 1.0))

            if ready:
                st.success(f"Ready to fine-tune ({thr}/class threshold met)")
            else:
                st.warning(f"Threshold: {thr} samples per class")

    if status:
        _render_status_card(col_s_audio, "audio", status["audio"])
        _render_status_card(col_s_image, "image", status["image"])

    st.divider()

    # ── Fine-tune controls ─────────────────────────────────────────────────
    st.subheader("Fine-Tune Models on Feedback Data")
    st.caption(
        "Fine-tuning loads existing model weights as a warm start, "
        "then trains only on your feedback data. "
        "Old weights are backed up automatically before any update."
    )

    col_ft_mode, col_ft_opts, col_ft_btn = st.columns([2, 2, 1])

    with col_ft_mode:
        retrain_mode = st.selectbox(
            "Modality to fine-tune",
            options=["both", "audio", "image"],
            help="Fine-tune audio model, image model, or both.",
        )
    with col_ft_opts:
        force_retrain = st.checkbox(
            "Force (ignore threshold check)",
            value=False,
            help="Run fine-tuning even if the dataset is below the minimum sample threshold.",
        )
    with col_ft_btn:
        st.write("")
        st.write("")
        run_retrain = st.button("Run Fine-Tune", type="primary", use_container_width=True)

    if run_retrain:
        with st.spinner("Fine-tuning in progress — this may take several minutes..."):
            try:
                ft_results = maybe_auto_retrain(mode=retrain_mode, force=force_retrain)
                st.success("Fine-tuning complete!")

                for mod, res in ft_results.items():
                    with st.expander(f"{mod.upper()} — result", expanded=True):
                        if res.get("skipped"):
                            st.warning(f"Skipped: {res['reason']}")
                        elif res.get("improved"):
                            old = res.get("val_acc_old", 0)
                            new = res.get("val_acc_new", 0)
                            st.success(
                                f"Model improved and saved.  "
                                f"Val accuracy: {old:.4f} → {new:.4f} "
                                f"(+{new - old:.4f})"
                            )
                        else:
                            old = res.get("val_acc_old", 0)
                            new = res.get("val_acc_new", 0)
                            st.info(
                                f"Trained but no improvement — old model kept.  "
                                f"Val accuracy: {old:.4f} → {new:.4f}"
                            )
            except Exception as exc:
                st.error(f"Fine-tuning failed: {exc}")
                st.exception(exc)

    st.divider()

    # ── Feedback log viewer ────────────────────────────────────────────────
    st.subheader("Feedback Log")
    log_csv = DATASET_ROOT / "feedback_log.csv"

    if log_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(log_csv)
            st.dataframe(df, use_container_width=True)

            col_dl, _ = st.columns([1, 3])
            with col_dl:
                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="feedback_log.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.caption(f"Could not load log: {exc}")
    else:
        st.caption("No feedback logged yet. Submit some labeled videos first.")

    # ── Dataset file counts ────────────────────────────────────────────────
    st.subheader("Dataset File Browser")
    for modality in ("audio", "frames"):
        mod_dir = DATASET_ROOT / modality
        if not mod_dir.exists():
            continue
        with st.expander(f"{modality.capitalize()} files"):
            for cls_dir in sorted(mod_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                files = list(cls_dir.iterdir())
                st.write(f"**{cls_dir.name}/** — {len(files)} file(s)")
