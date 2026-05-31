# Q2: English to Urdu translation

Two takes on the same problem. The first is a Transformer written from scratch to show the mechanics of attention based translation. The second is a fine tuned mBART-50, included as a strong baseline that actually produces fluent Urdu.

## Two tracks

### From scratch Transformer

A full Vaswani encoder decoder built without `nn.Transformer`. It uses a per side BPE subword tokeniser at 8k merges, Noam learning rate scheduling, label smoothing, and gradient clipping, with BLEU evaluation during training.

The corpus has 22,808 training pairs, which is small for translation, and the model mode collapses as a result. After 40 epochs, including 20 with EDA augmentation, BLEU sits around 0.43. This outcome is expected at this data scale and is documented honestly rather than hidden. Augmentation shifts the collapse phrase but does not resolve the underlying problem.

### mBART-50 fine tune

`facebook/mbart-large-50-many-to-many-mmt` used both zero shot and after a short 3 epoch fine tune. Both produce fluent Urdu on the test sentences, and the fine tuned version is a little more idiomatic. This is what a practical system would use, and it gives the from scratch model something honest to be compared against.

## Files

```
transformer_model.py    From scratch Vaswani encoder decoder, no nn.Transformer
data_utils.py           Corpus loader, train and val and test split, BPE training, dataset
augmentation.py         EDA from Wei and Zou 2019: random deletion, swap, and WordNet synonyms
transformer_train.py    Noam schedule, label smoothing, grad clipping, BLEU eval, --augment_prob, --resume
mbart_finetune.py       mBART-50 fine tune, using text_target= and tok.tgt_lang="ur_PK"
inference.py            Load a checkpoint and greedy decode
modal/q2_scratch_aug.py Modal L4 wrapper that runs the Transformer with --augment_prob 0.4
modal/q2_mbart.py       Modal A100 wrapper that runs the mBART fine tune
notebooks/              Colab and Kaggle notebooks used for the runs
```

## Reproduce

From scratch inference, which demonstrates the mode collapse:

```bash
pip install torch tokenizers sacrebleu
python inference.py --ckpt weights/latest.pt "How are you today?"
```

The Modal wrappers run the actual training. The scratch track runs 40 epochs with augmentation, and the mBART track runs 3 epochs on an A100. Checkpoints and the BPE vocabularies are written to a persistent volume.

## A note on Urdu rendering

Urdu is right to left and needs reshaping before it displays correctly. Anywhere the code renders Urdu text for figures, it reshapes with arabic-reshaper and applies the bidi algorithm first, otherwise the letters do not join. On Windows, Segoe UI carries the Arabic glyphs while some default fonts do not. See `../docs/report.pdf` for the full discussion.
