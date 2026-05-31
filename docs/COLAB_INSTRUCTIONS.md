# GPU Training on Google Colab

Training the three models locally on CPU is **not feasible** - CycleGAN and
mBART need a GPU, ViT benefits enormously. Every task has a ready Colab
notebook that does the GPU work. After training, weights come back to this
folder and the inference / Flask UI runs fine on CPU.

## General pattern

1. Go to https://colab.research.google.com/ (free T4 GPU tier is enough).
2. Upload the `.ipynb` for the task you want to train.
3. Runtime > **Change runtime type > GPU**.
4. Run the cells top-to-bottom. You will be prompted to:
   - upload `kaggle.json` (for dataset download),
   - upload the `.py` source files for that task.
5. Download the resulting weights back to this repo when done.

## Q1 - CycleGAN (Colab)

Notebook: `Q1_CycleGAN/colab_cyclegan.ipynb`

- Upload `kaggle.json`, `cyclegan_model.py`, `dataset.py`, `cyclegan_train.py`.
- The dataset unzips into folders whose names vary (`photos` vs `images`,
  `sketches` vs `pencil_sketches`). The notebook prints the tree; pass the
  right names with `--photos_dir`, `--sketches_dir`.
- Training runs ~100 epochs at batch size 1 on a T4 (about 8-10 hours total;
  set `--n_epochs 30` for a shorter demo run - results are still usable).
- **Resume**: `--resume` reloads `weights/latest.pt`. If the Colab session
  disconnects, re-run the same cell; training picks up at the last saved epoch.
- At the end, download `weights/generators.pt` and drop it into
  `Q1_CycleGAN/weights/`.

Sample previews are written every epoch to `samples/epoch_NNN.png` so you can
watch quality improve during training.

## Q2 - English -> Urdu Transformer (Colab)

Notebook: `Q2_Translation/colab_translation.ipynb`

- Upload `kaggle.json`, plus all 5 `.py` files in that folder.
- The scratch Transformer at `d_model=256, depth=4, heads=8` trains in 20-30
  min on a T4 for 30 epochs on the 24k-pair corpus. BLEU is logged after every
  epoch on a 200-sample validation subset.
- Download `weights_scratch.tar.gz`, extract to `Q2_Translation/weights/`, and
  run `python inference.py --ckpt weights/latest.pt "Hello world."` locally.

### Optional mBART baseline

The same notebook can fine-tune `facebook/mbart-large-50-many-to-many-mmt`.
Needs ~16 GB GPU RAM; Colab Pro / A100 is recommended. 3 epochs on 24k pairs
takes ~1-2 hours on a T4 at `batch_size=4`.

## Q3 - ViT vs CNN on CIFAR-10 (Colab)

Notebook: `Q3_ViT_CNN/colab_cifar.ipynb`

- No Kaggle needed - torchvision auto-downloads CIFAR-10.
- Upload `models.py`, `train.py`, `compare.py`.
- Typical wall times on T4:
  - CNN 50 epochs: ~15-20 min (~85-90 % val acc).
  - ViT 80 epochs: ~30-40 min (~75-80 % val acc from scratch on CIFAR).
  - Pretrained ViT-tiny 5 epochs: ~10 min (~95+ % val acc).
- `compare.py` merges all three into a single bar chart + overlay val-acc plot.
- Download `q3_results.tar.gz` and extract into `Q3_ViT_CNN/` to retain
  weights, JSON results, and plots for the report.

## Troubleshooting

- **Colab disconnects mid-training**: every task supports `--resume`. Re-run
  the same training cell; it reloads `weights/latest.pt`.
- **CUDA OOM on Q1**: drop `--batch_size` to 1 (default) and `--image_size` to
  128. Paper uses 256 but 128 is acceptable for demo.
- **Colab CIFAR download fails**: torchvision occasionally hits a bad mirror.
  Replace the download cell with `!pip install gdown` and pull from a known
  mirror, or pre-upload a CIFAR tarball.
- **Kaggle quota**: free tier allows ~20 GB / day. These datasets are <200 MB.
