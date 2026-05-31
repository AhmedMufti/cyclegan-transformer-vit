"""
Modal app for Q2 mBART-50 fine-tune on the parallel English-Urdu corpus.
Runs on A100-40GB, batch 4, 3 epochs. ~90 minutes.

Weights + logs persist to a Modal Volume across restarts.
"""
from pathlib import Path
import modal

app = modal.App("genai-q2-mbart")
vol = modal.Volume.from_name("genai-q2-mbart-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip", "git")
    .pip_install(
        "torch==2.5.1", "numpy",
        "transformers==4.46.0",
        "sentencepiece",
        "sacrebleu",
        "kaggle",
        "accelerate",
    )
)

LOCAL_Q2 = Path(__file__).parent.parent / "Q2_Translation"
image = image.add_local_file(LOCAL_Q2 / "data_utils.py",     "/root/data_utils.py")
image = image.add_local_file(LOCAL_Q2 / "mbart_finetune.py", "/root/mbart_finetune.py")


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("kaggle")],
    timeout=4 * 3600,
)
def train():
    import os, subprocess, sys
    from pathlib import Path

    log_dir = Path("/vol/logs");   log_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("/vol/data");  data_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path("/vol/weights_mbart"); weights_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "train.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # Dataset download
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

    # Find the Dataset folder (contains english-corpus.txt + urdu-corpus.txt)
    dataset_root = None
    for r, _, files in os.walk(data_dir):
        if any("english" in f.lower() for f in files) and any("urdu" in f.lower() for f in files):
            dataset_root = r; break
    if not dataset_root:
        raise RuntimeError(f"No parallel corpus found under {data_dir}")
    log(f"Using dataset_root={dataset_root}")

    cmd = [
        sys.executable, "/root/mbart_finetune.py",
        "--data_root", dataset_root,
        "--out_dir", str(weights_dir),
        "--epochs", "3",
        "--batch_size", "4",
        "--num_workers", "0",
        "--max_len", "128",
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
            if line.startswith("Saved "):
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
    for rel in ["weights_mbart", "logs"]:
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


@app.function(
    image=image, gpu="A100-40GB", volumes={"/vol": vol}, timeout=600,
)
def inference_demo():
    """Run a small inference on the final epoch and return the 5 translations."""
    from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
    import torch
    # Find latest epoch
    from pathlib import Path
    base = Path("/vol/weights_mbart")
    epochs = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("epoch_"))
    if not epochs:
        return "(no finetuned weights yet)"
    latest = epochs[-1]
    tok = MBart50TokenizerFast.from_pretrained(str(latest))
    model = MBartForConditionalGeneration.from_pretrained(str(latest)).cuda().eval()
    tok.src_lang = "en_XX"
    ur_bos = tok.lang_code_to_id["ur_PK"]
    out_lines = [f"=== Inference from {latest.name} ===", ""]
    sents = ["How are you today?",
             "I love reading books in the library.",
             "The weather is very nice this morning.",
             "She is a very good teacher.",
             "Education is the key to success."]
    with torch.no_grad():
        for s in sents:
            ids = tok(s, return_tensors="pt").to("cuda")
            gen = model.generate(**ids, forced_bos_token_id=ur_bos,
                                 max_length=128, num_beams=4)
            urdu = tok.decode(gen[0], skip_special_tokens=True)
            out_lines += [f"EN: {s}", f"UR: {urdu}", ""]
    return "\n".join(out_lines)
