"""
retrain.py — Retrain audio and/or image models on the accumulated feedback dataset.

The feedback system saves labelled samples into:
    dataset_creation/
        audio/  real/  fake/
        frames/ real/  fake/

Training scripts (Audio_Model_Training/Train.py and
Image_Model_Training/Train_with_logs.py) expect a train/val/test split:
    <split_root>/
        train/ real/ fake/
        val/   real/ fake/
        test/  real/ fake/

This script:
  1. Reads dataset_creation/{audio|frames}/{real,fake}/
  2. Shuffles and splits into train/val/test (default 70/15/15)
  3. Copies files into a temporary split directory
  4. Calls the appropriate training script
  5. Copies the best model back to Program_Workflow/ so main.py picks it up

Usage:
    # From Program_Workflow/ directory
    python retrain.py                          # retrain both audio + image
    python retrain.py --mode audio             # audio only
    python retrain.py --mode image             # image only
    python retrain.py --train-split 0.8 --val-split 0.1
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------

def _collect_files(class_dirs: dict[str, Path], exts: set[str]) -> dict[str, list[Path]]:
    """
    Returns {class_name: [file, ...]} for all files with matching extensions.
    """
    result: dict[str, list[Path]] = {}
    for cls, d in class_dirs.items():
        if not d.exists():
            print(f"[WARN] Class dir not found, skipping: {d}")
            result[cls] = []
            continue
        files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]
        result[cls] = sorted(files)
    return result


def _split_files(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split a list of files into train / val / test."""
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    train = shuffled[:n_train]
    val = shuffled[n_train: n_train + n_val]
    test = shuffled[n_train + n_val:]
    if not test:
        # if dataset is tiny give at least one sample to test from val
        test = val[-1:]
        val = val[:-1]
    return train, val, test


def _copy_split(
    class_splits: dict[str, tuple[list[Path], list[Path], list[Path]]],
    split_root: Path,
):
    """Write files into split_root/train|val|test/<class>/."""
    if split_root.exists():
        shutil.rmtree(split_root)
    for cls, (train, val, test) in class_splits.items():
        for subset_name, files in [("train", train), ("val", val), ("test", test)]:
            dst_dir = split_root / subset_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, dst_dir / src.name)
    print(f"[SPLIT] Written to: {split_root.resolve()}")


def _print_split_summary(
    class_splits: dict[str, tuple[list[Path], list[Path], list[Path]]],
    label: str,
):
    print(f"\n[{label}] Dataset split summary:")
    total = sum(len(tr) + len(va) + len(te) for tr, va, te in class_splits.values())
    for cls, (tr, va, te) in class_splits.items():
        print(f"  {cls}: {len(tr)} train, {len(va)} val, {len(te)} test")
    print(f"  Total: {total} files")


def _run(cmd: list[str], cwd: Path):
    print(f"\n[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        print(f"[ERROR] Command failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Retrain deepfake models on feedback dataset")
    parser.add_argument("--dataset-root", default="dataset_creation",
                        help="Feedback dataset root (default: dataset_creation)")
    parser.add_argument("--mode", choices=["audio", "image", "both"], default="both",
                        help="Which modality to retrain (default: both)")
    parser.add_argument("--train-split", type=float, default=0.70,
                        help="Fraction of data for training (default: 0.70)")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of data for validation (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    here = Path(__file__).parent.resolve()          # Program_Workflow/
    repo = here.parent.resolve()                    # project root
    dataset_root = (here / args.dataset_root).resolve()
    tmp_root = here / ".retrain_split"

    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    # ---- AUDIO ----
    if args.mode in ("audio", "both"):
        print("\n========== AUDIO RETRAINING ==========")
        audio_class_dirs = {
            "real": dataset_root / "audio" / "real",
            "fake": dataset_root / "audio" / "fake",
        }
        audio_files = _collect_files(audio_class_dirs, AUDIO_EXTS)
        total_audio = sum(len(v) for v in audio_files.values())

        if total_audio < 4:
            print(f"[SKIP] Not enough audio samples ({total_audio}). "
                  f"Need at least 4 (one per class per split). Run feedback first.")
        else:
            audio_splits = {
                cls: _split_files(files, args.train_split, args.val_split, args.seed)
                for cls, files in audio_files.items()
            }
            _print_split_summary(audio_splits, "AUDIO")

            audio_split_root = tmp_root / "audio"
            _copy_split(audio_splits, audio_split_root)

            audio_train_script = repo / "Audio_Model_Training" / "Train.py"
            audio_out_model = here / "best_audio_model.pt"

            _run([
                sys.executable, str(audio_train_script),
                "--data-root", str(audio_split_root),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--workers", str(args.workers),
                "--seed", str(args.seed),
                "--output", "best_audio_model.pt",
                "--results-dir", str(here / "audio_retrain_results"),
            ], cwd=here)

            # The training script saves the model inside its results dir;
            # also look for it directly in cwd (Train.py uses args.output as filename)
            candidate = here / "best_audio_model.pt"
            if candidate.exists():
                print(f"[AUDIO] Best model at: {candidate.resolve()}")
            else:
                print("[AUDIO] Model file not found at expected path — check results dir.")

    # ---- IMAGE ----
    if args.mode in ("image", "both"):
        print("\n========== IMAGE RETRAINING ==========")
        image_class_dirs = {
            "real": dataset_root / "frames" / "real",
            "fake": dataset_root / "frames" / "fake",
        }
        image_files = _collect_files(image_class_dirs, IMAGE_EXTS)
        total_image = sum(len(v) for v in image_files.values())

        if total_image < 4:
            print(f"[SKIP] Not enough image samples ({total_image}). "
                  f"Need at least 4 (one per class per split). Run feedback first.")
        else:
            image_splits = {
                cls: _split_files(files, args.train_split, args.val_split, args.seed)
                for cls, files in image_files.items()
            }
            _print_split_summary(image_splits, "IMAGE")

            image_split_root = tmp_root / "frames"
            _copy_split(image_splits, image_split_root)

            image_train_script = repo / "Image_Model_Training" / "Train_with_logs.py"
            image_out_model = here / "best_model.pt"

            _run([
                sys.executable, str(image_train_script),
                "--data-root", str(image_split_root),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--workers", str(args.workers),
                "--seed", str(args.seed),
                "--output", str(image_out_model),
                "--log-dir", str(here / "image_retrain_logs"),
            ], cwd=here)

            if image_out_model.exists():
                print(f"[IMAGE] Best model at: {image_out_model.resolve()}")
            else:
                print("[IMAGE] Model file not found at expected path — check log dir.")

    # Cleanup temporary split directory
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
        print(f"\n[CLEANUP] Removed temporary split dir: {tmp_root}")

    print("\nDone. Re-run main.py to use the updated models.")


if __name__ == "__main__":
    main()
