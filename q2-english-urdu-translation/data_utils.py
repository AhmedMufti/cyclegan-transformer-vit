"""
Data + BPE tokenizer utilities for the English-Urdu corpus.

The 'parallel-corpus-for-english-urdu-language' Kaggle dataset ships as CSV
or .txt files. We accept either shape and normalise to two aligned lists.
"""
import os
import random
import re
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


def _clean(s):
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_pairs(root):
    """Look for a CSV with 'English','Urdu' columns, else two parallel .txt files."""
    root = Path(root)
    csvs = list(root.glob("*.csv"))
    pairs = []

    if csvs:
        import csv
        with open(csvs[0], "r", encoding="utf-8", errors="ignore") as f:
            rdr = csv.DictReader(f)
            # tolerate different column names
            eng_key = next((k for k in rdr.fieldnames if "eng" in k.lower()), rdr.fieldnames[0])
            urd_key = next((k for k in rdr.fieldnames if "urd" in k.lower()), rdr.fieldnames[1])
            for row in rdr:
                en, ur = _clean(row.get(eng_key, "")), _clean(row.get(urd_key, ""))
                if en and ur:
                    pairs.append((en, ur))
    else:
        en_file = next((p for p in root.glob("*english*") if p.suffix in {".txt", ".en"}), None)
        ur_file = next((p for p in root.glob("*urdu*")    if p.suffix in {".txt", ".ur"}), None)
        if not en_file or not ur_file:
            raise FileNotFoundError(f"No CSV or parallel txt files found in {root}")
        with open(en_file, encoding="utf-8", errors="ignore") as f:
            en = [_clean(l) for l in f if l.strip()]
        with open(ur_file, encoding="utf-8", errors="ignore") as f:
            ur = [_clean(l) for l in f if l.strip()]
        pairs = list(zip(en, ur))
    return pairs


def split_pairs(pairs, val_ratio=0.05, test_ratio=0.02, seed=42):
    rng = random.Random(seed)
    idx = list(range(len(pairs))); rng.shuffle(idx)
    n = len(pairs)
    n_val = int(n * val_ratio); n_test = int(n * test_ratio)
    test = [pairs[i] for i in idx[:n_test]]
    val = [pairs[i] for i in idx[n_test:n_test + n_val]]
    train = [pairs[i] for i in idx[n_test + n_val:]]
    return train, val, test


def train_bpe(sentences, vocab_size=8000, save_path=None):
    """Train a BPE tokenizer on an iterable of strings."""
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL,
                         show_progress=False)
    tok.train_from_iterator(sentences, trainer)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tok.save(save_path)
    return tok


def load_bpe(path):
    return Tokenizer.from_file(path)


def encode(tok, s, add_bos=False, add_eos=False, max_len=128):
    ids = tok.encode(s).ids[: max_len - 2]
    if add_bos: ids = [BOS_IDX] + ids
    if add_eos: ids = ids + [EOS_IDX]
    return ids


class TranslationDataset(Dataset):
    """
    Optional on-the-fly augmentation on the source side (EDA, Wei & Zou 2019).
    `augment_prob=0` disables it (default) -> identical behaviour to before.
    """
    def __init__(self, pairs, src_tok, tgt_tok, max_len=128, augment_prob=0.0):
        self.pairs = pairs
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.max_len = max_len
        self.augment_prob = augment_prob
        if augment_prob > 0:
            # imported lazily so the default path has no new dependency
            from augmentation import augment_sentence
            self._augment = augment_sentence
        else:
            self._augment = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        en, ur = self.pairs[i]
        if self._augment is not None:
            en = self._augment(en, p_aug=self.augment_prob)
        src = encode(self.src_tok, en, add_bos=True, add_eos=True, max_len=self.max_len)
        tgt = encode(self.tgt_tok, ur, add_bos=True, add_eos=True, max_len=self.max_len)
        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch):
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=PAD_IDX)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=PAD_IDX)
    return src, tgt
