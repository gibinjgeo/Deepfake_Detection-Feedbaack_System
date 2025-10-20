#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFile

# --- GUI imports ---
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# Handle truncated/corrupted images gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --------------------- model utils ---------------------
def load_label_map(json_path: Path):
    with open(json_path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class

def load_checkpoint(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt.get("model_name", "resnet18")
    img_size = ckpt.get("img_size", 224)
    state_dict = ckpt["state_dict"]
    classes = ckpt.get("classes", None)  # optional

    # Build the backbone with the correct head size later
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        head_in = model.fc.in_features
        last = "fc"
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        head_in = model.classifier[1].in_features
        last = "classifier.1"
    else:
        raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")

    return model, model_name, img_size, head_in, last, state_dict, classes

def attach_head(model, last: str, head_in: int, num_classes: int):
    if last == "fc":
        model.fc = nn.Linear(head_in, num_classes)
    elif last == "classifier.1":
        model.classifier[1] = nn.Linear(head_in, num_classes)
    else:
        raise ValueError(f"Unknown last layer: {last}")
    return model

def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

# --------------------- prediction ---------------------
@torch.no_grad()
def predict_files(model, img_paths, tfm, device, idx_to_class):
    model.eval()
    results = []
    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            results.append((str(p), "ERROR", 0.0, f"Failed to open: {e}"))
            continue
        x = tfm(img).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        label = idx_to_class[pred_idx]
        conf = float(probs[pred_idx])
        results.append((str(p), label, conf, None))
    return results

# --------------------- GUI flow ---------------------
def main():
    # Paths (assumes files are beside this script; adjust if needed)
    ckpt_path = Path("best_model.pt")
    map_path  = Path("class_to_idx.json")

    if not ckpt_path.exists() or not map_path.exists():
        messagebox.showerror("Missing files",
                             "Could not find best_model.pt and/or class_to_idx.json "
                             "in the current directory.")
        return

    # Load mapping first to know number of classes
    class_to_idx, idx_to_class = load_label_map(map_path)
    num_classes = len(class_to_idx)

    # Build model from checkpoint metadata
    model, model_name, img_size, head_in, last, state_dict, _ = load_checkpoint(ckpt_path)
    model = attach_head(model, last, head_in, num_classes)
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tfm = build_transform(img_size)

    # Tk root
    root = tk.Tk()
    root.withdraw()  # hide main window

    # Ask how many images
    try:
        n = simpledialog.askinteger("Select count",
                                    "How many images do you want to classify?",
                                    minvalue=1, maxvalue=10000, parent=root)
    except Exception:
        n = None
    if not n:
        messagebox.showinfo("Canceled", "No number entered. Exiting.")
        return

    # File picker (multi-select)
    filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                 ("All files", "*.*")]
    paths = filedialog.askopenfilenames(title=f"Pick up to {n} images",
                                        filetypes=filetypes)
    if not paths:
        messagebox.showinfo("Canceled", "No images selected. Exiting.")
        return

    # Enforce n if user picked more
    paths = list(paths)[:n]

    # Predict
    results = predict_files(model, paths, tfm, device, idx_to_class)

    # Build a nice text report
    lines = []
    for p, label, conf, err in results:
        if err:
            lines.append(f"{p}\n  ERROR: {err}")
        else:
            lines.append(f"{p}\n  Prediction: {label}   Confidence: {conf:.4f}")
    report = "\n\n".join(lines)

    # Show results in a simple window
    result_win = tk.Toplevel()
    result_win.title("Predictions")
    text = tk.Text(result_win, wrap="word", height=30, width=100)
    text.insert("1.0", report)
    text.config(state="disabled")
    text.pack(padx=10, pady=10)

    # Offer to save CSV
    if messagebox.askyesno("Save CSV", "Do you want to save predictions to a CSV file?"):
        save_path = filedialog.asksaveasfilename(
            title="Save predictions",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="predictions.csv"
        )
        if save_path:
            import csv
            with open(save_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "prediction", "confidence", "error"])
                for p, label, conf, err in results:
                    w.writerow([p, label, f"{conf:.6f}" if conf else "", err or ""])
            messagebox.showinfo("Saved", f"Predictions saved to:\n{save_path}")

    # Also print to console
    print("\n=== Predictions ===")
    print(report)

    # Keep window open until closed
    result_win.mainloop()

if __name__ == "__main__":
    main()
