# Setup / Portability Guide

Instructions for getting this project running on a fresh laptop.

## 1. Python

- Python **3.10 - 3.12** (tested with 3.12.5 on Windows).
- From the project root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you have an NVIDIA GPU locally, install the CUDA-matched PyTorch first:

```bash
# e.g. CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

otherwise the CPU build is fine (training still runs on Colab).

## 2. Folder layout

```
genai assignment 2/
  Spring26_GenAI_Assignment_2.pdf
  requirements.txt
  SETUP.md          <- this file
  COLAB_INSTRUCTIONS.md
  DEMO_GUIDE.md
  GPT_PROMPTS.txt
  Q1_CycleGAN/
    cyclegan_model.py      generators + PatchGAN discriminator
    dataset.py             Person Face Sketches dataloader
    cyclegan_train.py      end-to-end training loop (+ resume)
    inference.py           load weights, auto-detect sketch vs face
    flask_app.py           Flask UI (upload / webcam)
    templates/index.html   UI front-end
    colab_cyclegan.ipynb   GPU training notebook
    weights/               drop generators.pt here for the UI
    samples/               epoch samples are written here during training
  Q2_Translation/
    transformer_model.py   Vaswani transformer from scratch
    data_utils.py          BPE tokenizers + dataset
    transformer_train.py   Noam LR, label smoothing, BLEU eval
    inference.py           load ckpt, translate
    mbart_finetune.py      optional pretrained baseline
    colab_translation.ipynb
    weights/               bpe_en.json, bpe_ur.json, latest.pt
  Q3_ViT_CNN/
    models.py              SimpleCNN (ResNet-ish) + ViT (from scratch)
    train.py               train/eval, curves, confusion matrix, report
    compare.py             merge all 3 results into a table + plots
    colab_cifar.ipynb
    weights/ plots/
```

## 3. Datasets

Training is meant to run on **Colab GPU** (see `COLAB_INSTRUCTIONS.md`). If you
want to pull the data locally:

| Task | Source | Path expected by code |
|------|--------|-----------------------|
| Q1 | Kaggle `almightyj/person-face-sketches` | `Q1_CycleGAN/data/photos/`, `Q1_CycleGAN/data/sketches/` (exact folder names may differ after unzip - pass `--photos_dir` / `--sketches_dir`) |
| Q2 | Kaggle `zainuddin123/parallel-corpus-for-english-urdu-language` | `Q2_Translation/data/` (any CSV with English/Urdu columns, or two parallel .txt) |
| Q3 | torchvision auto-downloads | `Q3_ViT_CNN/data/` |

To pull from Kaggle locally:

1. From kaggle.com -> account -> "Create New API Token", save `kaggle.json` into `~/.kaggle/`.
2. `pip install kaggle`
3. `kaggle datasets download -d <slug> -p <target_folder> && unzip ...`

## 4. Moving the folder to a new laptop

The project is **self-contained** inside `genai assignment 2/`. To move:

1. Copy the entire directory (weights included if you have them).
2. On the new laptop, redo the venv step above. Do not check the venv into Git; it's platform-specific.
3. Datasets are **not** checked in - re-download with Kaggle or via the Colab notebooks.
4. Paths in the code are all relative. No absolute paths need updating.

### If you already trained weights on Colab

- Q1 UI needs `Q1_CycleGAN/weights/generators.pt`.
- Q2 inference needs `Q2_Translation/weights/{latest.pt, bpe_en.json, bpe_ur.json}`.
- Q3 evaluation needs `Q3_ViT_CNN/weights/<model>/best.pt`.

Drop the files into those folders after moving the project - nothing else to change.

## 5. Running a demo without a GPU

You can **evaluate and run the Flask UI on CPU**. Inference of CycleGAN on a
single 256x256 image takes ~2 s on CPU; translation is similarly fast; CIFAR
inference is instant. Only *training* needs a GPU.

## 6. Known dependency quirks

- On Windows, `num_workers > 0` in DataLoader requires running training from `if __name__ == "__main__":` (all scripts here already do this).
- `torchvision` >= 0.16 is required for the newer CIFAR-10 URL.
- `tokenizers` wheels are pre-built for Python 3.10-3.12; 3.13 may fail to install.
- `timm` is optional (only needed for `--model pretrained` in Q3).
