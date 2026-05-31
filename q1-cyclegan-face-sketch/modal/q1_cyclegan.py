"""
Modal app for Q1 CycleGAN - train from scratch at paper-faithful 256x256, 30
epochs, H100. Weights + samples are written to a persistent Modal Volume so
they survive any restart.

Usage:
    modal run -d modal_apps/q1_cyclegan.py::train        # launch detached
    modal run modal_apps/q1_cyclegan.py::tail_log        # peek latest log
    modal run modal_apps/q1_cyclegan.py::download        # pull artifacts local

The container embeds the three source files directly so nothing extra has to
be uploaded.
"""
from pathlib import Path
import modal

app = modal.App("genai-q1-cyclegan")

# Persistent storage that survives any restart or disconnect.
vol = modal.Volume.from_name("genai-q1-cyclegan-vol", create_if_missing=True)

# Image: PyTorch with CUDA 12.4, plus Kaggle + Pillow for dataset download.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip", "git")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy",
        "pillow",
        "matplotlib",
        "tqdm",
        "kaggle",
        gpu="any",
    )
)

# -----------------------------------------------------------------------------
# Embed the CycleGAN source files inside the container. This keeps the Modal
# app self-contained (no extra uploads, no volume-version coupling).
# -----------------------------------------------------------------------------
LOCAL_Q1 = Path(__file__).parent.parent / "Q1_CycleGAN"
image = image.add_local_file(LOCAL_Q1 / "cyclegan_model.py", "/root/cyclegan_model.py")
image = image.add_local_file(LOCAL_Q1 / "dataset.py",       "/root/dataset.py")
image = image.add_local_file(LOCAL_Q1 / "cyclegan_train.py", "/root/cyclegan_train.py")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("kaggle")],
    timeout=6 * 3600,  # 6h hard cap
)
def train():
    """Run CycleGAN training from scratch at 256x256, 30 epochs.

    Re-runs of this function are idempotent: if weights/latest.pt is already
    in the Volume, training resumes instead of starting over. Logs are
    streamed to stdout AND mirrored to /vol/logs/train.log so they can be
    tailed later.
    """
    import os, subprocess, sys
    from pathlib import Path

    # Kaggle creds from the modal secret
    os.environ.setdefault("KAGGLE_USERNAME", os.environ["KAGGLE_USERNAME"])
    os.environ.setdefault("KAGGLE_KEY",      os.environ["KAGGLE_KEY"])

    log_dir = Path("/vol/logs");     log_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("/vol/data");    data_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path("/vol/weights"); weights_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = Path("/vol/samples"); samples_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "train.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
            f.flush()

    # ------------------------------------------------------------------
    # 1. Download the Person Face Sketches dataset (once).
    # ------------------------------------------------------------------
    marker = data_dir / ".download_complete"
    if not marker.exists():
        log("=== Downloading Person Face Sketches dataset ===")
        subprocess.check_call([
            "kaggle", "datasets", "download",
            "-d", "almightyj/person-face-sketches",
            "-p", str(data_dir), "--unzip",
        ])
        marker.touch()
        vol.commit()
    else:
        log("=== Dataset already present in Volume ===")

    # Discover the photos/sketches folders (Kaggle repacks vary).
    photos_root, sketches_root = None, None
    for root, _, _ in os.walk(data_dir):
        base = Path(root).name
        if base == "photos" and (Path(root).parent / "sketches").exists():
            photos_root = Path(root).parent
            break
    if not photos_root:
        for root, dirs, _ in os.walk(data_dir):
            if "photos" in dirs and "sketches" in dirs:
                photos_root = Path(root); break
    if not photos_root:
        raise RuntimeError(f"No photos/sketches folders under {data_dir}")
    log(f"Using data_root={photos_root}")

    # ------------------------------------------------------------------
    # 2. Launch training.
    # ------------------------------------------------------------------
    cmd = [
        sys.executable, "/root/cyclegan_train.py",
        "--data_root", str(photos_root),
        "--photos_dir", "photos",
        "--sketches_dir", "sketches",
        "--weights_dir", str(weights_dir),
        "--samples_dir", str(samples_dir),
        "--image_size", "256",
        "--batch_size", "8",
        "--num_workers", "4",
        "--n_epochs", "30",
        "--decay_epoch", "15",
        "--log_every", "100",
        "--resume",   # picks up latest.pt if present
    ]
    log(f"=== Launching: {' '.join(cmd)} ===")

    # Stream child output line by line, mirror to log file, commit Volume
    # after each epoch so data isn't lost on a crash.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1, cwd="/root")
    try:
        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
                f.flush()
            # Commit Volume when an epoch wraps up
            if "Epoch" in line and "done" in line:
                try: vol.commit()
                except Exception as e: print(f"[vol.commit warn] {e}")
    finally:
        ret = proc.wait()
        try: vol.commit()
        except Exception: pass
    log(f"=== Training finished with exit code {ret} ===")
    return ret


@app.function(image=image, volumes={"/vol": vol})
def tail_log(lines: int = 200):
    """Return the last N lines of the training log."""
    from pathlib import Path
    p = Path("/vol/logs/train.log")
    if not p.exists():
        return "(no log yet)"
    txt = p.read_text().splitlines()
    return "\n".join(txt[-lines:])


@app.function(image=image, volumes={"/vol": vol})
def list_artifacts():
    """Return a dict of important files + sizes in the Volume."""
    import os
    from pathlib import Path
    out = {}
    for rel in ["weights", "samples", "logs"]:
        p = Path(f"/vol/{rel}")
        if p.exists():
            out[rel] = [(f.name, f.stat().st_size) for f in p.iterdir() if f.is_file()]
    return out


@app.function(image=image, volumes={"/vol": vol})
def read_file(path: str) -> bytes:
    """Return raw bytes of any file under /vol."""
    with open(path, "rb") as f:
        return f.read()
