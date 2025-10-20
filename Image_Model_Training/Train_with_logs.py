# Train_with_logs.py  (torch.amp version)
import argparse, csv, json, os, random, signal, sys, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Optional metrics (binary)
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# --- Robust image IO & perf knobs
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ----------------- small logging helpers -----------------
class Logger:
    def __init__(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filepath, "w", buffering=1)
        self.start = time.time()
        self.log("=" * 80)
        self.log(f"Log file: {filepath.resolve()}")
        self.log(f"Start time: {datetime.now().isoformat()}")
        self.log("=" * 80)

    def log(self, s: str = ""):
        print(s)
        try:
            self.file.write(s + "\n")
            self.file.flush()
            os.fsync(self.file.fileno())
        except Exception:
            pass

    def close(self):
        elapsed = time.time() - self.start
        self.log(f"\nTotal elapsed: {elapsed:.2f}s")
        self.log("=" * 80)
        try:
            self.file.close()
        except Exception:
            pass

class CsvWriter:
    def __init__(self, path: Path, header: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        self.header = header

    def write(self, row: dict):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.header).writerow(row)

# ----------------- dataset & transforms -----------------
class SafeImageFolder(datasets.ImageFolder):
    """Won't crash on a bad image; uses a blank fallback."""
    def __init__(self, *args, img_size=224, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = img_size
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except Exception:
            sample = Image.new('RGB', (self.img_size, self.img_size))
        if self.transform is not None: sample = self.transform(sample)
        if self.target_transform is not None: target = self.target_transform(target)
        return sample, target

def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

def _resolve_split(root: Path, primary: str, fallback: str) -> Path:
    p1, p2 = root/primary, root/fallback
    if p1.exists(): return p1
    if p2.exists(): return p2
    raise FileNotFoundError(f"Could not find '{primary}' or '{fallback}' under {root}")

def load_datasets(data_root: Path, img_size: int, logger_print):
    tr, ev = build_transforms(img_size)
    train_root = _resolve_split(data_root, "train", "training")
    val_root   = _resolve_split(data_root, "val", "validation")
    test_root  = _resolve_split(data_root, "test", "testing")
    train_ds = SafeImageFolder(str(train_root), transform=tr, img_size=img_size)
    val_ds   = SafeImageFolder(str(val_root),   transform=ev, img_size=img_size)
    test_ds  = SafeImageFolder(str(test_root),  transform=ev, img_size=img_size)
    with open("class_to_idx_img.json", "w") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)
    logger_print(f"Classes: {train_ds.classes}")
    logger_print(f"Training samples: {len(train_ds)} | Validation: {len(val_ds)} | Testing: {len(test_ds)}")
    return train_ds, val_ds, test_ds

# ----------------- models -----------------
def build_model(num_classes: int, backbone: str):
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return m

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------- metrics helpers -----------------
def confusion_matrix(y_true, y_pred, k):
    cm = np.zeros((k,k), dtype=int)
    for t,p in zip(y_true, y_pred): cm[t,p]+=1
    return cm

def macro_prf_from_cm(cm):
    P,R,F = [],[],[]
    for c in range(cm.shape[0]):
        tp = cm[c,c]
        fp = cm[:,c].sum()-tp
        fn = cm[c,:].sum()-tp
        p = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f = (2*p*r/(p+r)) if (p+r)>0 else 0.0
        P.append(p); R.append(r); F.append(f)
    return float(np.mean(P)), float(np.mean(R)), float(np.mean(F)), P,R,F

# ----------------- globals to export epoch/LR into eval logger -----------------
current_epoch = [0]
current_lr    = [0.0]

# ----------------- evaluation (per-batch timing + optional mem) -----------------
@torch.no_grad()
def evaluate(model, loader, device, criterion=None, need_probs=False,
             phase="val", batch_writer: CsvWriter|None=None, mem_stats=False, use_amp=False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    y_true, y_pred, y_scores = [], [], []

    start = time.time()
    prev_end = start
    for bidx, (imgs, labels) in enumerate(loader):
        data_time = time.time() - prev_end
        t0 = time.time()
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp and device.type=="cuda"):
            logits = model(imgs)
            loss_v = criterion(logits, labels).item() if criterion is not None else None
        fwd_time = time.time() - t0

        preds = logits.argmax(1)
        correct += (preds==labels).sum().item()
        total   += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        if need_probs and logits.size(1)==2:
            probs = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
            y_scores.extend(probs.tolist())
            prob_mean = float(probs.mean())
        else:
            prob_mean = ""

        batch_time = data_time + fwd_time
        imgs_per_s = imgs.size(0) / batch_time if batch_time>0 else 0.0
        mem_alloc = mem_resv = ""
        if mem_stats and torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated()/(1024**2)
            mem_resv  = torch.cuda.memory_reserved() /(1024**2)

        if criterion is not None: loss_sum += loss_v * imgs.size(0)

        if batch_writer is not None:
            batch_writer.write({
                "phase": phase, "epoch": current_epoch[0], "batch_idx": bidx,
                "batch_size": imgs.size(0), "lr": current_lr[0],
                "loss": loss_v if criterion is not None else "",
                "acc": (preds==labels).float().mean().item(),
                "data_time_s": data_time, "fwd_time_s": fwd_time,
                "bwd_time_s": "", "opt_time_s": "",
                "batch_time_s": batch_time, "imgs_per_s": imgs_per_s,
                "grad_norm": "", "prob_mean_cls1": prob_mean,
                "gpu_mem_alloc_MB": mem_alloc, "gpu_mem_reserved_MB": mem_resv
            })
        prev_end = time.time()

    avg_loss = (loss_sum/total) if (criterion is not None and total) else None
    acc = correct/total if total else 0.0
    dur = time.time() - start
    return avg_loss, acc, np.array(y_true), np.array(y_pred), (np.array(y_scores) if need_probs else None), dur

# ----------------- interrupt-safe checkpoint -----------------
def _setup_interrupt_handler(state_callable, logger):
    def handler(signum, frame):
        logger.log("\n[INTERRUPT] saving last_interrupt_img_checkpoint.pt ...")
        try:
            torch.save(state_callable(), "last_interrupt_img_checkpoint.pt")
        except Exception as e:
            logger.log(f"Failed to save interrupt checkpoint: {e}")
        finally:
            logger.close(); sys.exit(130)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

# ----------------- main training -----------------
def train(args):
    # logging & seeds
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(log_dir / "image_training_results.txt"); lg = logger.log
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lg(f"Device: {device} | PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available(): lg(f"GPU name: {torch.cuda.get_device_name(0)}")
    lg(f"Args: {vars(args)}")

    # data
    train_ds, val_ds, test_ds = load_datasets(Path(args.data_root), args.img_size, lg)

    pin = (device.type == "cuda")
    pw  = bool(args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin,
                              persistent_workers=pw, prefetch_factor=2 if pw else None)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin,
                              persistent_workers=pw, prefetch_factor=2 if pw else None)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin,
                              persistent_workers=pw, prefetch_factor=2 if pw else None)

    # model/opt
    num_classes = len(train_ds.classes)
    model = build_model(num_classes, args.backbone).to(device)
    lg(f"Model: {args.backbone} | Trainable params: {count_trainable_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # writers
    epoch_writer = CsvWriter(log_dir/"epoch_log.csv",
        ["epoch","phase","loss","acc","lr","epoch_time_s","eval_time_s",
         "early_stop_triggered","best_val_acc","best_epoch"])
    batch_writer = CsvWriter(log_dir/"batch_log.csv",
        ["phase","epoch","batch_idx","batch_size","lr","loss","acc",
         "data_time_s","fwd_time_s","bwd_time_s","opt_time_s","batch_time_s",
         "imgs_per_s","grad_norm","prob_mean_cls1","gpu_mem_alloc_MB","gpu_mem_reserved_MB"])

    class History:
        def __init__(self, csv_path: Path, json_path: Path):
            self.csv_path, self.json_path, self.rows = csv_path, json_path, []
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr","note"])
        def add(self, e, trl, tra, vall, vala, lr, note=""):
            self.rows.append({"epoch": e,"train_loss": float(trl),"train_acc": float(tra),
                              "val_loss": float(vall),"val_acc": float(vala),"lr": float(lr),"note": note})
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([e, trl, tra, vall, vala, lr, note])
        def finalize(self, meta: dict):
            with open(self.json_path, "w") as f:
                json.dump({"metadata": meta, "history": self.rows}, f, indent=2)

    hist_csv, hist_json = log_dir/"training_history_img.csv", log_dir/"training_history_img.json"
    history = History(hist_csv, hist_json)

    best_val_acc, best_epoch = 0.0, -1
    patience_left, early_stop = args.early_stop, False
    wall_start, time_of_best = time.time(), None

    # interrupt snapshot
    def snapshot():
        return {"model_name": args.backbone, "img_size": args.img_size,
                "state_dict": model.state_dict(), "classes": train_ds.classes,
                "best_val_acc": best_val_acc, "best_epoch": best_epoch}
    _setup_interrupt_handler(snapshot, logger)

    # training epochs
    global current_epoch, current_lr
    for epoch in range(1, args.epochs+1):
        current_epoch[0] = epoch
        model.train()
        run_loss = run_correct = run_total = 0
        epoch_t0, prev_end = time.time(), time.time()

        for bidx, (imgs, labels) in enumerate(train_loader):
            data_time = time.time() - prev_end
            t0 = time.time()
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # AMP forward
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            fwd_time = time.time() - t0

            # backward + step
            t1 = time.time()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bwd_time = time.time() - t1

            preds = logits.argmax(1)
            run_loss   += loss.item() * imgs.size(0)
            run_correct+= (preds==labels).sum().item()
            run_total  += labels.size(0)

            batch_time = data_time + fwd_time + bwd_time
            imgs_per_s = imgs.size(0) / batch_time if batch_time>0 else 0.0
            lr_now     = optimizer.param_groups[0]['lr']
            current_lr[0] = lr_now

            mem_alloc = mem_resv = ""
            if args.mem_stats and torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated()/(1024**2)
                mem_resv  = torch.cuda.memory_reserved() /(1024**2)

            batch_writer.write({
                "phase":"train","epoch":epoch,"batch_idx":bidx,"batch_size":imgs.size(0),
                "lr":lr_now,"loss":loss.item(),"acc":(preds==labels).float().mean().item(),
                "data_time_s":data_time,"fwd_time_s":fwd_time,"bwd_time_s":bwd_time,
                "opt_time_s":0.0,"batch_time_s":batch_time,"imgs_per_s":imgs_per_s,
                "grad_norm":"","prob_mean_cls1":"","gpu_mem_alloc_MB":mem_alloc,"gpu_mem_reserved_MB":mem_resv
            })
            prev_end = time.time()

        train_loss = run_loss / run_total if run_total else 0.0
        train_acc  = run_correct / run_total if run_total else 0.0
        epoch_time = time.time() - epoch_t0

        # validation
        need_probs = (num_classes == 2)
        val_loss, val_acc, *_ , val_dur = evaluate(
            model, val_loader, device, criterion, need_probs=need_probs,
            phase="val", batch_writer=batch_writer, mem_stats=args.mem_stats, use_amp=use_amp
        )
        scheduler.step(val_acc)
        lr_now = optimizer.param_groups[0]['lr']

        logger.log(f"Epoch {epoch:02d}/{args.epochs} | "
                   f"Train {train_loss:.4f}/{train_acc:.4f} | "
                   f"Val {val_loss:.4f}/{val_acc:.4f} | LR {lr_now:.2e} | "
                   f"epoch {epoch_time:.1f}s | val_eval {val_dur:.1f}s")

        epoch_writer.write({
            "epoch": epoch, "phase": "train+val",
            "loss": f"{train_loss:.6f}/{val_loss:.6f}",
            "acc": f"{train_acc:.6f}/{val_acc:.6f}",
            "lr": lr_now, "epoch_time_s": epoch_time,
            "eval_time_s": val_dur, "early_stop_triggered": "",
            "best_val_acc": "", "best_epoch": ""
        })
        history.add(epoch, train_loss, train_acc, val_loss, val_acc, lr_now)

        # checkpoint / early stop logic
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(snapshot(), args.output)
            if time_of_best is None: time_of_best = time.time() - wall_start
            logger.log(f"✅ Saved BEST to {args.output} (best_val_acc={best_val_acc:.4f}, epoch={best_epoch})")
            patience_left = args.early_stop
        else:
            patience_left -= 1
            logger.log(f"No improvement. Early-stop patience left: {patience_left}")
            if patience_left == 0:
                early_stop = True
                logger.log("⛔ Early stopping triggered.")
                epoch_writer.write({
                    "epoch": epoch, "phase": "early_stop",
                    "loss": "", "acc": "", "lr": lr_now,
                    "epoch_time_s": epoch_time, "eval_time_s": val_dur,
                    "early_stop_triggered": True,
                    "best_val_acc": best_val_acc, "best_epoch": best_epoch
                })
                break

    # rollback to best & test
    logger.log("\nRolling back to best checkpoint for final evaluation...")
    ckpt = torch.load(args.output, map_location="cpu")
    logger.log(f"Rolled back to epoch: {ckpt.get('best_epoch','?')}, best_val_acc={ckpt.get('best_val_acc', None)}")
    model = build_model(num_classes, args.backbone).to(device)
    model.load_state_dict(ckpt["state_dict"])

    need_probs_test = (num_classes == 2)
    test_loss, test_acc, y_true, y_pred, y_scores, test_dur = evaluate(
        model, test_loader, device, criterion, need_probs=need_probs_test,
        phase="test", batch_writer=batch_writer, mem_stats=args.mem_stats, use_amp=use_amp
    )
    cm = confusion_matrix(y_true, y_pred, num_classes)
    mp, mr, mf1, per_p, per_r, per_f1 = macro_prf_from_cm(cm)

    logger.log("\n=== TEST RESULTS ===")
    logger.log(f"Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | test_eval {test_dur:.1f}s")
    logger.log(f"Macro Precision: {mp:.4f} | Macro Recall: {mr:.4f} | Macro F1: {mf1:.4f}")
    for i, cls in enumerate(train_ds.classes):
        logger.log(f"Class '{cls}': P={per_p[i]:.4f} R={per_r[i]:.4f} F1={per_f1[i]:.4f}")
    logger.log("\nConfusion matrix (rows=true, cols=pred):\n" + str(cm))

    extra = {}
    if need_probs_test and (y_scores is not None) and SKLEARN_OK:
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
            ap      = average_precision_score(y_true, y_scores)
            extra = {"roc_auc": float(roc_auc), "average_precision": float(ap)}
            logger.log(f"\nExtra (binary): ROC-AUC={roc_auc:.4f} | Average Precision={ap:.4f}")
        except Exception as e:
            logger.log(f"[WARN] Could not compute ROC/AUPRC: {e}")
    elif need_probs_test and not SKLEARN_OK:
        logger.log("[INFO] sklearn not available; skipping ROC-AUC/AUPRC.")

    metadata = {
        "datetime_start": datetime.fromtimestamp(wall_start).isoformat(),
        "datetime_end": datetime.now().isoformat(),
        "wall_clock_total_s": time.time() - wall_start,
        "time_to_best_s": time_of_best,
        "args": vars(args),
        "device": str(device),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "classes": train_ds.classes, "num_classes": num_classes,
        "best_epoch": ckpt.get("best_epoch", None), "best_val_acc": float(ckpt.get("best_val_acc", 0.0)),
        "early_stop_triggered": early_stop,
        "final_test": {
            "loss": float(test_loss) if test_loss is not None else None,
            "accuracy": float(test_acc),
            "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf1),
            "per_class_precision": [float(x) for x in per_p],
            "per_class_recall": [float(x) for x in per_r],
            "per_class_f1": [float(x) for x in per_f1],
            "confusion_matrix": cm.tolist(),
            **extra
        }
    }
    with open(log_dir/"training_history_img.json", "w") as f:
        json.dump({"metadata": metadata, "history": []}, f, indent=2)

    logger.log(f"\nSaved per-epoch CSV: {log_dir/'training_history_img.csv'}")
    logger.log(f"Saved epoch log CSV: {epoch_writer.path}")
    logger.log(f"Saved batch log CSV: {batch_writer.path}")
    logger.log(f"Saved full metadata JSON: {log_dir/'training_history_img.json'}")
    logger.close()

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Train real-vs-fake image classifier with deep telemetry (torch.amp)")
    p.add_argument("--data-root", type=str, default=".", help="Folder that contains train|training, val|validation, test|testing")
    p.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["resnet18","efficientnet_b0"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stop", type=int, default=4, help="patience epochs without val acc improvement")
    p.add_argument("--output", type=str, default="best_image_model.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", type=str, default="logs_img", help="Directory to store logs/CSVs/JSON")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP)")
    p.add_argument("--mem-stats", action="store_true", help="Log CUDA memory per batch")
    return p.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        train(args)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        raise
