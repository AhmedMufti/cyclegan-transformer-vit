"""
Train a CNN and a ViT on CIFAR-10, save per-epoch metrics, plot curves,
confusion matrix, and example predictions.

Run:
    python train.py --model cnn --epochs 50
    python train.py --model vit --epochs 50
    python train.py --model pretrained --epochs 5   # (uses timm ViT if available)

Metrics + plots go to ./plots/<model>/
Weights saved every epoch to ./weights/<model>/
"""
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SimpleCNN, ViT, count_params


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def build_transforms(img_size=32, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(img_size) if img_size != 32 else transforms.Lambda(lambda x: x),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return transforms.Compose([
        transforms.Resize(img_size) if img_size != 32 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def build_model(name, num_classes=10):
    if name == "cnn":
        return SimpleCNN(num_classes), 32
    if name == "vit":
        return ViT(img_size=32, patch_size=4, num_classes=num_classes,
                   d_model=192, depth=6, n_heads=6), 32
    if name == "pretrained":
        import timm
        model = timm.create_model("vit_tiny_patch16_224", pretrained=True,
                                  num_classes=num_classes)
        return model, 224
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0; correct = 0; y_true = []; y_pred = []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        out = model(x)
        total_loss += loss_fn(out, y).item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        y_true.extend(y.cpu().tolist()); y_pred.extend(pred.cpu().tolist())
    return total_loss / total, correct / total, y_true, y_pred


def plot_history(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = list(range(1, len(history["train_loss"]) + 1))
    ax1.plot(ep, history["train_loss"], label="train"); ax1.plot(ep, history["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(ep, history["train_acc"], label="train"); ax2.plot(ep, history["val_acc"], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def plot_confusion(y_true, y_pred, out_path):
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Confusion matrix")
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def save_examples(model, loader, device, out_path, n=16):
    model.eval()
    x, y = next(iter(loader))
    x_in = x[:n].to(device)
    with torch.no_grad():
        pred = model(x_in).argmax(1).cpu()
    x_show = x[:n] * torch.tensor(CIFAR_STD)[:, None, None] + torch.tensor(CIFAR_MEAN)[:, None, None]
    x_show = x_show.clamp(0, 1)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(x_show[i].permute(1, 2, 0).numpy())
        ok = pred[i].item() == y[i].item()
        ax.set_title(f"p:{CLASSES[pred[i]]}\nt:{CLASSES[y[i]]}",
                     color="green" if ok else "red", fontsize=8)
        ax.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "vit", "pretrained"], required=True)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--weights_dir", type=str, default="./weights")
    p.add_argument("--plots_dir", type=str, default="./plots")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weights_dir = Path(args.weights_dir) / args.model; weights_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir) / args.model; plots_dir.mkdir(parents=True, exist_ok=True)

    model, img_size = build_model(args.model)
    model = model.to(device)
    print(f"Model: {args.model} ({count_params(model):,} params) @ {img_size}x{img_size}")

    train_ds = datasets.CIFAR10(args.data_root, train=True, download=True,
                                transform=build_transforms(img_size, train=True))
    test_ds = datasets.CIFAR10(args.data_root, train=False, download=True,
                               transform=build_transforms(img_size, train=False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0; bad = 0; start_epoch = 0

    ckpt_path = weights_dir / "latest.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); history = ck["history"]
        best_acc = ck["best_acc"]; start_epoch = ck["epoch"] + 1
        print(f"Resumed epoch {start_epoch}")

    total_train_time = 0.0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time(); model.train()
        ep_loss = 0.0; ep_correct = 0; ep_total = 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * y.size(0)
            ep_correct += (out.argmax(1) == y).sum().item()
            ep_total += y.size(0)
        sched.step()

        train_loss = ep_loss / ep_total; train_acc = ep_correct / ep_total
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, device)
        dt = time.time() - t0; total_train_time += dt
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc); history["val_acc"].append(val_acc)
        print(f"ep {epoch} train_loss {train_loss:.3f} train_acc {train_acc:.3f} "
              f"val_loss {val_loss:.3f} val_acc {val_acc:.3f} time {dt:.1f}s")

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "history": history, "best_acc": best_acc},
                   ckpt_path)

        if val_acc > best_acc:
            best_acc = val_acc; bad = 0
            torch.save(model.state_dict(), weights_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val_acc={best_acc:.3f})")
                break

    # Reload best weights for final eval
    best_path = weights_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    _, final_acc, y_true, y_pred = evaluate(model, val_loader, device)

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    results = {
        "model": args.model, "params": count_params(model),
        "best_val_acc": float(final_acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "total_train_time_sec": float(total_train_time),
        "epochs_run": len(history["train_loss"]),
        "history": history,
    }
    with open(plots_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_history(history, plots_dir / "loss_acc_curves.png")
    plot_confusion(y_true, y_pred, plots_dir / "confusion_matrix.png")
    save_examples(model, val_loader, device, plots_dir / "example_preds.png")
    with open(plots_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=CLASSES))

    print(f"\nFinished. best_val_acc={final_acc:.4f}  f1={f1:.4f}  "
          f"params={count_params(model):,}  time={total_train_time:.1f}s")
    print(f"Artefacts in {plots_dir}")


if __name__ == "__main__":
    main()
