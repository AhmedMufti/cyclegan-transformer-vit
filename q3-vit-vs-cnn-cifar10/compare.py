"""
Aggregate results from CNN / ViT / pretrained runs into a single comparison
table + bar charts.

Run AFTER training all three models:
    python train.py --model cnn --epochs 50
    python train.py --model vit --epochs 50
    python train.py --model pretrained --epochs 5
    python compare.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(plots_dir="./plots"):
    rows = {}
    for sub in Path(plots_dir).iterdir():
        rj = sub / "results.json"
        if rj.exists():
            rows[sub.name] = json.loads(rj.read_text())
    return rows


def main():
    rows = load_results()
    if not rows:
        print("No results found. Run train.py for at least one model first.")
        return

    # Table
    header = f"{'model':<12}{'params':>12}{'val_acc':>10}{'f1':>8}{'time(s)':>10}{'epochs':>8}"
    print(header); print("-" * len(header))
    for name, r in rows.items():
        print(f"{name:<12}{r['params']:>12,}{r['best_val_acc']:>10.4f}"
              f"{r['f1_macro']:>8.3f}{r['total_train_time_sec']:>10.1f}"
              f"{r['epochs_run']:>8}")

    # Accuracy + parameter count bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    names = list(rows.keys())
    ax1.bar(names, [rows[n]["best_val_acc"] for n in names], color="steelblue")
    ax1.set_title("Best val accuracy"); ax1.set_ylim(0, 1)
    ax2.bar(names, [rows[n]["params"] / 1e6 for n in names], color="indianred")
    ax2.set_title("Parameters (M)")
    plt.tight_layout(); plt.savefig("plots/comparison.png", dpi=120); plt.close()

    # Overlay training curves
    plt.figure(figsize=(8, 4))
    for n in names:
        plt.plot(rows[n]["history"]["val_acc"], label=f"{n} val")
    plt.title("Validation accuracy per epoch"); plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/val_acc_curves.png", dpi=120); plt.close()
    print("\nSaved plots/comparison.png + plots/val_acc_curves.png")


if __name__ == "__main__":
    main()
