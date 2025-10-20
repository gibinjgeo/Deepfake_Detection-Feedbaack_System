#!/usr/bin/env python3
"""
Build a single, balanced, training-ready audio dataset from multiple sources.

It scans the following (recursively):
- archive/*/*/(training|validation|testing)/(fake|real)
- Sample_Dataset/(fake|real)
- Train/(training|validation|testing)/(fake|real)
- Train/Outside_Eval/DeepFake Dataset/(All_Fake_Audios|All_Real_Audios)  [treated as unsplit]

Output (default): Unified_Audio_Dataset/
  train/{fake,real}  val/{fake,real}  test/{fake,real}

Balancing: Stratified 80/10/10; equal counts per class (minority-limited).
De-dup: content-hash (md5) + size. Uses hardlinks if possible, else copies.

Supported extensions: .wav .mp3 .flac .ogg .m4a .aac
"""

import argparse
import hashlib
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Dict, List, Tuple

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# ----------------------- helpers -----------------------

def is_audio(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in AUDIO_EXTS

def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def try_hardlink_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)  # hardlink = instant if same filesystem
    except Exception:
        shutil.copy2(src, dst)  # fallback to copy

def gather_by_class_from_root(root: Path) -> Dict[str, List[Path]]:
    """
    Given a root that contains subfolders named exactly the class names (fake/real),
    return a dict { 'fake': [...], 'real': [...] } collecting audio files.
    Missing class folders are tolerated.
    """
    out = {"fake": [], "real": []}
    for cls in ["fake", "real"]:
        cdir = root / cls
        if cdir.exists() and cdir.is_dir():
            for p in cdir.rglob("*"):
                if is_audio(p):
                    out[cls].append(p)
    return out

def collect_all_sources(base_dir: Path) -> Dict[str, List[Path]]:
    """
    Crawl all the places you showed and return ALL audio paths grouped by class,
    ignoring any existing split labels (we will re-split globally).
    """
    collected = {"fake": [], "real": []}

    # 1) archive/*/*/(training|validation|testing)/(fake|real)
    archive = base_dir / "archive"
    if archive.exists():
        for variant in archive.iterdir():
            if not variant.is_dir():
                continue
            # the inner folder (e.g., for-2sec/for-2seconds)
            for inner in variant.iterdir():
                if not inner.is_dir():
                    continue
                for split_name in ["training", "validation", "testing", "train", "val", "test"]:
                    split_dir = inner / split_name
                    if split_dir.exists():
                        by_cls = gather_by_class_from_root(split_dir)
                        collected["fake"].extend(by_cls["fake"])
                        collected["real"].extend(by_cls["real"])

    # 2) Sample_Dataset/(fake|real)  (unsplit — we’ll include and re-split)
    sample_ds = base_dir / "Sample_Dataset"
    if sample_ds.exists():
        by_cls = gather_by_class_from_root(sample_ds)
        collected["fake"].extend(by_cls["fake"])
        collected["real"].extend(by_cls["real"])

    # 3) Train/(training|validation|testing)/(fake|real)
    train_root = base_dir / "Train"
    if train_root.exists():
        for split_name in ["training", "validation", "testing", "train", "val", "test"]:
            split_dir = train_root / split_name
            if split_dir.exists():
                by_cls = gather_by_class_from_root(split_dir)
                collected["fake"].extend(by_cls["fake"])
                collected["real"].extend(by_cls["real"])

        # 4) Train/Outside_Eval/DeepFake Dataset/(All_Fake_Audios|All_Real_Audios)
        deepfake_ds = train_root / "Outside_Eval" / "DeepFake Dataset"
        if deepfake_ds.exists():
            fake_dir = deepfake_ds / "All_Fake_Audios"
            real_dir = deepfake_ds / "All_Real_Audios"
            if fake_dir.exists():
                collected["fake"].extend([p for p in fake_dir.rglob("*") if is_audio(p)])
            if real_dir.exists():
                collected["real"].extend([p for p in real_dir.rglob("*") if is_audio(p)])

    return collected

def deduplicate(paths: List[Path]) -> List[Path]:
    """
    Deduplicate by (size, md5). Faster than md5-only on large sets.
    """
    by_size: Dict[int, List[Path]] = {}
    for p in paths:
        try:
            sz = p.stat().st_size
        except FileNotFoundError:
            continue
        by_size.setdefault(sz, []).append(p)

    unique: List[Path] = []
    seen_hashes: set = set()
    for sz, group in by_size.items():
        if len(group) == 1:
            unique.append(group[0])
            continue
        # same size — hash to confirm duplicates
        local_seen = {}
        for p in group:
            try:
                h = md5sum(p)
            except Exception:
                # if unreadable, skip it
                continue
            if h in seen_hashes:
                continue
            if h in local_seen:
                # duplicate within this size-group: keep just the first
                continue
            local_seen[h] = p
            seen_hashes.add(h)
            unique.append(p)
    return unique

def stratified_split(
    items_by_class: Dict[str, List[Path]],
    ratios=(0.8, 0.1, 0.1),
    seed: int = 42
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Re-split globally into train/val/test with equal counts per class.
    Returns: { 'train': {'fake': [...], 'real': [...]}, 'val': {...}, 'test': {...} }
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    rng = random.Random(seed)

    classes = sorted(items_by_class.keys())
    # shuffle each class list deterministically
    for cls in classes:
        rng.shuffle(items_by_class[cls])

    # target total per class = limited by minority count
    min_count = min(len(items_by_class[cls]) for cls in classes)
    if min_count == 0:
        raise RuntimeError("One of the classes has zero items after collection; cannot balance.")

    n_train = int(min_count * ratios[0])
    n_val   = int(min_count * ratios[1])
    n_test  = min_count - n_train - n_val

    out = {
        "train": {cls: [] for cls in classes},
        "val":   {cls: [] for cls in classes},
        "test":  {cls: [] for cls in classes},
    }

    for cls in classes:
        pool = items_by_class[cls]
        out["train"][cls] = pool[0:n_train]
        out["val"][cls]   = pool[n_train:n_train+n_val]
        out["test"][cls]  = pool[n_train+n_val:n_train+n_val+n_test]

    return out

def materialize(output_root: Path, split_map: Dict[str, Dict[str, List[Path]]]):
    """
    Create folder tree and hardlink/copy files into place.
    """
    for split in ["train", "val", "test"]:
        for cls in ["fake", "real"]:
            safe_makedirs(output_root / split / cls)

    for split, cls_map in split_map.items():
        for cls, paths in cls_map.items():
            dst_dir = output_root / split / cls
            for src in paths:
                dst = dst_dir / src.name
                # If filename collision, add a numeric suffix
                if dst.exists():
                    stem = dst.stem
                    suf = 1
                    while True:
                        cand = dst_dir / f"{stem}__{suf}{dst.suffix}"
                        if not cand.exists():
                            dst = cand
                            break
                        suf += 1
                try_hardlink_or_copy(src, dst)

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Combine and balance audio datasets into a unified train/val/test layout.")
    ap.add_argument("--base-dir", type=str, default=".", help="Folder that contains archive/, Sample_Dataset/, Train/")
    ap.add_argument("--out", type=str, default="Unified_Audio_Dataset", help="Output dataset root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    args = ap.parse_args()

    if round(args.train_ratio + args.val_ratio + args.test_ratio, 6) != 1.0:
        print("ERROR: ratios must sum to 1.0", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(args.base_dir).resolve()
    out_root = Path(args.out).resolve()

    print(f"[INFO] Scanning sources under: {base_dir}")
    collected = collect_all_sources(base_dir)

    print(f"[INFO] Before de-dup: fake={len(collected['fake'])}, real={len(collected['real'])}")
    collected["fake"] = deduplicate(collected["fake"])
    collected["real"] = deduplicate(collected["real"])
    print(f"[INFO] After  de-dup: fake={len(collected['fake'])}, real={len(collected['real'])}")

    # Stratified, balanced re-split
    split_map = stratified_split(
        items_by_class=collected,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed
    )
    # Stats
    for split in ["train", "val", "test"]:
        f_cnt = len(split_map[split]["fake"])
        r_cnt = len(split_map[split]["real"])
        print(f"[INFO] {split:>5}: fake={f_cnt} | real={r_cnt} | total={f_cnt+r_cnt}")

    # Materialize
    if out_root.exists():
        print(f"[WARN] Output exists: {out_root}. Files may be added alongside existing.")
    print(f"[INFO] Writing to: {out_root}")
    materialize(out_root, split_map)
    print("[DONE] Unified, balanced dataset created.")

if __name__ == "__main__":
    main()
