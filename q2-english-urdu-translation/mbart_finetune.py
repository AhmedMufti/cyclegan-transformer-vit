"""
Optional: Fine-tune mBART-large-50 for English -> Urdu translation.

Strong baseline for comparison against the from-scratch Transformer.
Requires a GPU runtime (~16 GB RAM). See COLAB_INSTRUCTIONS.md.

Usage:
    python mbart_finetune.py --data_root /content/data --epochs 3
"""
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (MBart50TokenizerFast, MBartForConditionalGeneration,
                          get_linear_schedule_with_warmup)

from data_utils import load_pairs, split_pairs


MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"


class MBartPairs(Dataset):
    def __init__(self, pairs, tokenizer, max_len=128):
        self.pairs = pairs; self.tok = tokenizer; self.max_len = max_len
        # Source language = English, target = Urdu
        self.tok.src_lang = "en_XX"
        self.tgt_lang = "ur_PK"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        en, ur = self.pairs[i]
        # Required by MBart50TokenizerFast's text_target= internals:
        self.tok.src_lang = "en_XX"
        self.tok.tgt_lang = "ur_PK"
        enc = self.tok(text=en, text_target=ur,
                       max_length=self.max_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        labels = enc["labels"]
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    # Force Urdu decoder start
    model.config.forced_bos_token_id = tok.lang_code_to_id["ur_PK"]

    pairs = load_pairs(args.data_root)
    train_pairs, val_pairs, _ = split_pairs(pairs)
    train_ds = MBartPairs(train_pairs, tok, args.max_len)
    val_ds = MBartPairs(val_pairs, tok, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if i % 20 == 0:
                print(f"ep {epoch} it {i}/{len(train_loader)} loss {out.loss.item():.3f}")

        # sample predictions
        model.eval()
        with torch.no_grad():
            sample = next(iter(val_loader))
            gen = model.generate(
                sample["input_ids"].to(device),
                attention_mask=sample["attention_mask"].to(device),
                forced_bos_token_id=tok.lang_code_to_id["ur_PK"],
                max_length=args.max_len, num_beams=4,
            )
            for i in range(min(3, gen.size(0))):
                print("pred:", tok.decode(gen[i], skip_special_tokens=True))

        save_dir = Path(args.out_dir) / f"epoch_{epoch}"
        model.save_pretrained(save_dir); tok.save_pretrained(save_dir)
        print(f"Saved {save_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./weights/mbart")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
