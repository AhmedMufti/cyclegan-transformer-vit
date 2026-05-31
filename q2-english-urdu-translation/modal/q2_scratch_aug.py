"""
Modal app for Q2 from-scratch Transformer WITH EDA data augmentation enabled.

Runs on L4 (cheap), 20 epochs with --augment_prob 0.4. ~30 minutes.
Evidence for rubric item 2.6 (Data Augmentation, 7 marks).
"""
from pathlib import Path
import modal

app = modal.App("genai-q2-scratch-aug")
vol = modal.Volume.from_name("genai-q2-scratch-aug-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip", "git")
    .pip_install(
        "torch==2.5.1", "numpy",
        "tokenizers",
        "sacrebleu",
        "kaggle",
        "nltk",
    )
)

LOCAL_Q2 = Path(__file__).parent.parent / "Q2_Translation"
for fn in ["transformer_model.py", "data_utils.py", "transformer_train.py",
           "inference.py", "augmentation.py"]:
    image = image.add_local_file(LOCAL_Q2 / fn, f"/root/{fn}")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("kaggle")],
    timeout=2 * 3600,
)
def train():
    import os, subprocess, sys
    from pathlib import Path

    log_dir = Path("/vol/logs");    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("/vol/data");   data_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path("/vol/weights_aug"); weights_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "train.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f: f.write(msg + "\n")

    # Download WordNet so synonym-replacement works inside the container
    subprocess.call([sys.executable, "-c", "import nltk; nltk.download('wordnet', quiet=True)"])

    marker = data_dir / ".download_complete"
    if not marker.exists():
        log("=== Downloading parallel-corpus-for-english-urdu-language ===")
        subprocess.check_call([
            "kaggle", "datasets", "download",
            "-d", "zainuddin123/parallel-corpus-for-english-urdu-language",
            "-p", str(data_dir), "--unzip",
        ])
        marker.touch()
        vol.commit()

    # Locate the parallel txt folder
    dataset_root = None
    for r, _, files in os.walk(data_dir):
        if any("english" in f.lower() for f in files) and any("urdu" in f.lower() for f in files):
            dataset_root = r; break
    if not dataset_root:
        raise RuntimeError("corpus not found")
    log(f"Using dataset_root={dataset_root}")

    cmd = [
        sys.executable, "/root/transformer_train.py",
        "--data_root", dataset_root,
        "--weights_dir", str(weights_dir),
        "--epochs", "40",
        "--batch_size", "64",
        "--d_model", "256", "--n_heads", "8", "--n_layers", "4",
        "--d_ff", "1024",
        "--warmup", "2000",
        "--augment_prob", "0.4",
        "--resume",
    ]
    log(f"=== Launching: {' '.join(cmd)} ===")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1, cwd="/root")
    try:
        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
            if line.startswith("ep ") and "BLEU" in line:
                try: vol.commit()
                except Exception: pass
    finally:
        ret = proc.wait()
        try: vol.commit()
        except Exception: pass
    log(f"=== Training finished with exit code {ret} ===")
    return ret


@app.function(image=image, volumes={"/vol": vol})
def tail_log(lines: int = 200):
    from pathlib import Path
    p = Path("/vol/logs/train.log")
    if not p.exists(): return "(no log yet)"
    return "\n".join(p.read_text().splitlines()[-lines:])


@app.function(image=image, volumes={"/vol": vol})
def list_artifacts():
    import os
    out = {}
    for rel in ["weights_aug", "logs"]:
        p = f"/vol/{rel}"
        if os.path.exists(p):
            items = []
            for root, _, files in os.walk(p):
                for f in files:
                    full = os.path.join(root, f)
                    items.append((os.path.relpath(full, "/vol"), os.path.getsize(full)))
            out[rel] = items
    return out


@app.function(image=image, volumes={"/vol": vol})
def read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
