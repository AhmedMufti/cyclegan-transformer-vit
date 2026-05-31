# Q3: Vision Transformer vs CNN on CIFAR-10

A controlled comparison of three image classifiers on CIFAR-10: a convolutional network, a Vision Transformer trained from scratch, and a pretrained ViT-tiny fine tuned on the same data. The point is to show where attention helps, where it does not, and how much pretraining changes the picture.

## Models

- CNN, a ResNet-20 style network built from scratch.
- ViT from scratch, with 4x4 patches and 6 transformer layers, written patch embedding upward.
- Pretrained `vit_tiny_patch16_224` from `timm`, fine tuned for 3 epochs.

The CNN and the scratch ViT are deliberately matched at roughly 2.7 million parameters so the comparison is about architecture rather than capacity.

## Results

| Model | Parameters | Val accuracy | Macro F1 | Time | Epochs |
|-------|-----------:|-------------:|---------:|-----:|-------:|
| CNN, ResNet-20 like | 2,777,674 | 0.9278 | 0.928 | 31 min | 50 |
| ViT from scratch | 2,693,578 | 0.8219 | 0.821 | 36 min | 55 |
| Pretrained ViT-tiny | 5,526,346 | 0.9715 | 0.972 | 10 min | 3 |

The story is data efficiency. With only 50k images, the convolutional inductive bias wins clearly over an equally sized Transformer. Once the Transformer arrives already knowing what images look like, three epochs of fine tuning are enough to pass both.

## Files

```
models.py    SimpleCNN, the from scratch ViT, and a parameter counter
train.py     Shared training loop for --model {cnn,vit,pretrained}, with AdamW,
             cosine LR, label smoothing, and early stopping. Saves curves,
             confusion matrix, classification report, results.json, and a
             grid of example predictions
compare.py   Merges the three results.json files into a summary table plus
             accuracy and parameter bar charts and a val accuracy overlay
notebooks/   Self contained Colab and Kaggle notebooks that run all three
             models and then the comparison
```

## Reproduce

```bash
pip install torch torchvision timm seaborn scikit-learn matplotlib
python train.py --model cnn        --epochs 50 --batch_size 128 --lr 3e-4
python train.py --model vit        --epochs 80 --batch_size 128 --lr 5e-4
python train.py --model pretrained --epochs  3 --batch_size  64 --lr 1e-4
python compare.py
```

CIFAR-10 downloads automatically on first run. All three models trained on Colab for the reported numbers. The plots and confusion matrices that go into the report are in `../screenshots/Q3_ViT_CNN/`.
