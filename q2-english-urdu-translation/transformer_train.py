"""
Train the from-scratch Transformer on the English-Urdu parallel corpus.

Features:
- BPE tokenizers trained on each side.
- Noam-style LR warmup (d_model^-0.5 * min(step^-0.5, step*warmup^-1.5)).
- Label smoothing 0.1.
- Gradient clipping.
- Checkpoint every epoch + resume support.
- BLEU evaluation on a validation split using sacrebleu.

Typical run on Colab GPU:
    python transformer_train.py --data_root /content/data --epochs 30
"""
import argparse
import math
import os
import time
from pathlib import Path

import sacrebleu
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import (BOS_IDX, EOS_IDX, PAD_IDX, TranslationDataset,
                        collate_fn, load_pairs, split_pairs, train_bpe)
from transformer_model import Transformer, count_params


def noam_lr(step, d_model, warmup):
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def label_smoothed_nll(logits, target, smoothing=0.1, ignore_index=PAD_IDX):
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    smooth = -logp.mean(dim=-1)
    loss = (1 - smoothing) * nll + smoothing * smooth
    mask = (target != ignore_index).float()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def greedy_decode(model, src, max_len, device, bos=BOS_IDX, eos=EOS_IDX):
    model.eval()
    src_mask = model.make_src_mask(src)
    enc = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), bos, dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys)
        out = model.decode(ys, enc, src_mask, tgt_mask)
        nxt = out[:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
        if (nxt == eos).all():
            break
    return ys


def detokenize(tok, ids):
    ids = [i for i in ids if i not in (PAD_IDX, BOS_IDX, EOS_IDX)]
    return tok.decode(ids)


def evaluate_bleu(model, loader, tgt_tok, device, max_len):
    hyps, refs = [], []
    for src, tgt in loader:
        src = src.to(device)
        pred = greedy_decode(model, src, max_len, device)
        for p, t in zip(pred.tolist(), tgt.tolist()):
            hyps.append(detokenize(tgt_tok, p))
            refs.append(detokenize(tgt_tok, t))
    return sacrebleu.corpus_bleu(hyps, [refs]).score, hyps[:5], refs[:5]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    weights_dir = Path(args.weights_dir); weights_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    pairs = load_pairs(args.data_root)
    print(f"Loaded {len(pairs)} parallel sentence pairs.")
    train_pairs, val_pairs, test_pairs = split_pairs(pairs)
    print(f"train {len(train_pairs)} / val {len(val_pairs)} / test {len(test_pairs)}")

    # --- Tokenizers ---
    src_tok_path = weights_dir / "bpe_en.json"
    tgt_tok_path = weights_dir / "bpe_ur.json"
    if src_tok_path.exists() and tgt_tok_path.exists() and args.resume:
        from data_utils import load_bpe
        src_tok, tgt_tok = load_bpe(str(src_tok_path)), load_bpe(str(tgt_tok_path))
        print("Loaded existing BPE tokenizers.")
    else:
        src_tok = train_bpe((en for en, _ in train_pairs),
                            vocab_size=args.vocab_size, save_path=str(src_tok_path))
        tgt_tok = train_bpe((ur for _, ur in train_pairs),
                            vocab_size=args.vocab_size, save_path=str(tgt_tok_path))

    src_vocab = src_tok.get_vocab_size(); tgt_vocab = tgt_tok.get_vocab_size()
    print(f"Vocab sizes: en={src_vocab}, ur={tgt_vocab}")

    train_ds = TranslationDataset(train_pairs, src_tok, tgt_tok,
                                  max_len=args.max_len,
                                  augment_prob=args.augment_prob)
    val_ds = TranslationDataset(val_pairs, src_tok, tgt_tok, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    # --- Model ---
    model = Transformer(src_vocab, tgt_vocab,
                        d_model=args.d_model, n_heads=args.n_heads,
                        n_enc=args.n_layers, n_dec=args.n_layers,
                        d_ff=args.d_ff, dropout=args.dropout,
                        max_len=args.max_len, pad_idx=PAD_IDX).to(device)
    print(f"Model params: {count_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    start_epoch = 0; step = 0
    ckpt_path = weights_dir / "latest.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1; step = ck["step"]
        print(f"Resumed from epoch {start_epoch}, step {step}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time(); model.train(); total = 0.0; count = 0
        for i, (src, tgt) in enumerate(train_loader):
            src = src.to(device); tgt = tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

            logits = model(src, tgt_in)
            loss = label_smoothed_nll(logits, tgt_out, smoothing=0.1)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            step += 1
            for g in opt.param_groups:
                g["lr"] = args.lr_scale * noam_lr(step, args.d_model, args.warmup)
            opt.step()

            total += loss.item(); count += 1
            if i % args.log_every == 0:
                print(f"ep {epoch} it {i}/{len(train_loader)} "
                      f"loss {loss.item():.3f} lr {opt.param_groups[0]['lr']:.2e}")

        print(f"ep {epoch} avg_loss {total / max(1, count):.3f} time {time.time() - t0:.1f}s")

        # BLEU on a random 200-sample subset to stay fast
        val_loader_small = DataLoader(
            torch.utils.data.Subset(val_ds, list(range(min(200, len(val_ds))))),
            batch_size=args.batch_size, collate_fn=collate_fn)
        bleu, sample_hyps, sample_refs = evaluate_bleu(model, val_loader_small, tgt_tok,
                                                      device, args.max_len)
        print(f"ep {epoch} BLEU(val 200)={bleu:.2f}")
        for h, r in zip(sample_hyps[:3], sample_refs[:3]):
            print(f"  hyp: {h}\n  ref: {r}\n")

        torch.save({"epoch": epoch, "step": step,
                    "model": model.state_dict(), "opt": opt.state_dict(),
                    "args": vars(args),
                    "src_vocab": src_vocab, "tgt_vocab": tgt_vocab},
                   ckpt_path)
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "args": vars(args),
                    "src_vocab": src_vocab, "tgt_vocab": tgt_vocab},
                   weights_dir / f"epoch_{epoch:02d}.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--weights_dir", type=str, default="./weights")
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--lr_scale", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--augment_prob", type=float, default=0.0,
                   help="Probability of applying EDA augmentation to a training "
                        "sample (0 disables). Typical value: 0.3-0.5.")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
