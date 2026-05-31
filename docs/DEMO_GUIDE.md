# Demo Guide - Ace the Viva

One page per question, covering: what was built, the design choices you
should highlight, and the exact commands to run during the demo.

---

## Q1 - CycleGAN for Face <-> Sketch Translation

### What's implemented

- **Generator** (`cyclegan_model.py::GeneratorResNet`) - ResNet-based, 9
  residual blocks, reflection padding, InstanceNorm. This matches the paper's
  spec for 256x256 inputs.
- **Discriminator** (`cyclegan_model.py::Discriminator`) - 70x70 PatchGAN.
- **Training** (`cyclegan_train.py`) - LSGAN adversarial loss (MSE), cycle
  loss (L1, lambda=10), identity loss (L1, lambda=5), 50-image replay buffer,
  Adam(lr=2e-4, beta=(0.5,0.999)), linear LR decay after epoch 50, weights
  saved every epoch with resume support.
- **Inference + UI** (`inference.py`, `flask_app.py`, `templates/index.html`)
  - Auto-detects sketch vs face by checking how far from grayscale the input
    is; users can override the mode.
  - Accepts file upload **or** live webcam capture.
  - Returns a base64 PNG back to the browser.

### Talking points the examiner will probe

1. **Why two generators and two discriminators?**
   Because image-to-image translation between unpaired domains A and B is
   ill-posed. Cycle-consistency (`||G_BA(G_AB(a)) - a|| + ||G_AB(G_BA(b)) - b||`)
   locks the mapping down by forcing round-trip identity.
2. **Why LSGAN (MSE) instead of BCE?** Lower gradient saturation near the
   optimum, less mode collapse, the paper reports better visual quality.
3. **What does identity loss do?** Encourages G_BA(a) ~ a when `a` is already
   a real face. Keeps colour/tone stable when the input is already in the
   target domain.
4. **Why a PatchGAN discriminator?** Scores 70x70 patches instead of the
   whole image, reducing parameter count and producing sharper textures.
5. **Why InstanceNorm, not BatchNorm?** Better for style-transfer-style
   tasks; normalisation is per-sample, so stylistic statistics aren't mixed
   across the batch.
6. **Auto-detection heuristic:** sketches have very low per-pixel deviation
   from the grayscale average (`<12` on a 0-255 scale). That suffices for the
   demo; a proper classifier could replace it.

### Demo run

```bash
# (assumes weights/generators.pt from Colab has been dropped in)
cd Q1_CycleGAN
python flask_app.py --weights weights/generators.pt
# open http://127.0.0.1:5000
```

- Drag in a real face -> output should be a pencil sketch.
- Drag in a sketch (or click "Use webcam" -> a drawing) -> real-ish face.
- Toggle the mode dropdown to show the auto-detect result differs.

---

## Q2 - English -> Urdu Machine Translation

### What's implemented

- **Transformer from scratch** (`transformer_model.py`) - exact Vaswani
  architecture: sinusoidal positional encoding, multi-head scaled dot-product
  attention, post-norm residual blocks, encoder-decoder with causal+padding
  masks. **No `nn.Transformer` shortcut** - every component is visible.
- **BPE tokenizers** (`data_utils.py`) - subword BPE (Byte-Pair Encoding),
  one trained per language, vocab 8k, handles OOV in both English and Urdu
  script.
- **Training** (`transformer_train.py`) - Adam(0.9, 0.98, eps=1e-9), Noam
  LR schedule (warmup 2000, `d_model^-0.5 * min(step^-0.5, step*warmup^-1.5)`),
  label smoothing = 0.1, gradient clipping = 1.0, BLEU evaluation each epoch
  via sacrebleu.
- **mBART baseline** (`mbart_finetune.py`) - fine-tunes
  `facebook/mbart-large-50-many-to-many-mmt` for the same task.

### Talking points

1. **Why Transformer, not RNN?** Parallelisable training; attention handles
   long-range dependencies without the vanishing gradient problem.
2. **Why sinusoidal positional encoding?** Deterministic and length-extrapolable.
   Learned embeddings also work; sinusoidal is what the paper used.
3. **Why label smoothing + Noam LR?** Both come from the paper - smoothing
   reduces overconfidence and improves BLEU, Noam stabilises early training by
   a linear warmup before decaying.
4. **Why BPE?** Urdu has rich morphology; word-level tokenisation blows up
   the vocabulary. BPE gives a fixed-size subword vocab that handles any word.
5. **Why evaluate with BLEU?** Standard MT metric; sacrebleu gives a
   reproducible, corpus-level score with tokenisation handled.
6. **When would you prefer mBART?** Data-scarce language pairs - it leverages
   pretraining on 50 languages so 24k pairs can fine-tune a much stronger
   model than you could train from scratch.

### Demo run

```bash
cd Q2_Translation
python inference.py --ckpt weights/latest.pt \
  "How are you today?" \
  "I love reading books in the library." \
  "The weather is very nice this morning."
```

Show the BLEU number printed during training (logged per epoch).

---

## Q3 - Vision Transformer vs CNN on CIFAR-10

### What's implemented

- **Custom CNN** (`models.py::SimpleCNN`) - ResNet-20-style: stem conv +
  3 stages of 2 BasicBlocks each (64 -> 128 -> 256), global avg pool,
  dropout, fully-connected head. ~2.7 M params.
- **ViT from scratch** (`models.py::ViT`) - 4x4 patches on 32x32 -> 64
  tokens + 1 [CLS]. d_model=192, depth=6, heads=6, MLP ratio 4. Learned
  positional embedding. Pre-norm transformer blocks. ~2.7 M params (matched
  budget with the CNN for a fair comparison).
- **Pretrained ViT** - `vit_tiny_patch16_224` via `timm`, fine-tuned on
  CIFAR-10 (images upscaled to 224).
- **Training** (`train.py`) - AdamW + cosine LR + label smoothing 0.1, random
  crop + horizontal flip, early stopping on val accuracy, per-epoch
  checkpointing + resume, confusion matrix, classification report.
- **Comparison** (`compare.py`) - merges all results into a single table and
  two bar charts.

### Talking points

1. **Why is ViT-from-scratch weaker than the CNN on CIFAR-10?** ViT has no
   inductive bias for locality/translation-equivariance. The original paper
   is explicit: ViT beats CNNs only at ImageNet scale (~14 M+ images).
   CIFAR-10 is too small.
2. **What closes the gap?** Pretraining. Fine-tuning a pretrained ViT-tiny
   easily beats both from-scratch models (expected >95 % val acc in 5 epochs).
3. **Why 4x4 patches?** 32/4 = 8 -> 64 tokens + CLS = 65 total. Smaller
   patches give finer granularity; large patches would mean too few tokens for
   a 32x32 image.
4. **Why matched parameter budget (~2.7 M each)?** So the comparison isolates
   the architectural difference rather than model capacity.
5. **Why cosine LR schedule?** Standard practice for transformer training;
   smoother than step decay and the paper-aligned choice.
6. **Expected outcome of the run:**
   | Model         | params   | val acc (typical) |
   |---------------|----------|-------------------|
   | CNN           | ~2.7 M   | 85-90 %           |
   | ViT scratch   | ~2.7 M   | 70-80 %           |
   | ViT pretrained| ~5.7 M   | 95-97 %           |

### Demo run

If you want live numbers during the viva, run the CNN (fastest): a single
epoch on a CPU takes ~2 min and hits ~50 % val acc, enough to show the
pipeline. Full numbers come from Colab training.

```bash
cd Q3_ViT_CNN
python train.py --model cnn --epochs 50
python train.py --model vit --epochs 80
python train.py --model pretrained --epochs 5
python compare.py
```

Plots are saved in `plots/cnn/`, `plots/vit/`, `plots/pretrained/`, and
aggregated charts in `plots/comparison.png`, `plots/val_acc_curves.png`.

---

## Final touches for the demo

- **Show the code, not just the output.** Examiners often ask "where is the
  attention computed" - open `transformer_model.py::MultiHeadAttention`.
  "Where is cycle loss?" - `cyclegan_train.py`, inside `train()` where
  `loss_cycle` is computed.
- **Know the loss/metric numbers.** Print them during the demo.
- **If they ask about failures:** "CycleGAN sketches can collapse to
  pencil-like noise early on - the cycle and identity losses stabilise it.
  For Urdu, BPE on too-small vocab starts generating repeated tokens; we use
  8k which works well for 24k pairs."
