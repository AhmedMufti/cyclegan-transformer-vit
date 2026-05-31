"""
Load a trained Transformer checkpoint and translate English -> Urdu.

Usage:
    python inference.py --ckpt weights/latest.pt "How are you today?"
"""
import argparse
from pathlib import Path

import torch

from data_utils import BOS_IDX, EOS_IDX, PAD_IDX, load_bpe, encode
from transformer_model import Transformer


def load_model(ckpt_path, bpe_dir=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device)
    a = ck["args"]
    model = Transformer(ck["src_vocab"], ck["tgt_vocab"],
                        d_model=a["d_model"], n_heads=a["n_heads"],
                        n_enc=a["n_layers"], n_dec=a["n_layers"],
                        d_ff=a["d_ff"], dropout=0.0,
                        max_len=a["max_len"], pad_idx=PAD_IDX).to(device).eval()
    model.load_state_dict(ck["model"])

    bpe_dir = Path(bpe_dir or Path(ckpt_path).parent)
    src_tok = load_bpe(str(bpe_dir / "bpe_en.json"))
    tgt_tok = load_bpe(str(bpe_dir / "bpe_ur.json"))
    return model, src_tok, tgt_tok, device, a["max_len"]


@torch.no_grad()
def translate(model, src_tok, tgt_tok, device, sentence, max_len=128):
    ids = encode(src_tok, sentence, add_bos=True, add_eos=True, max_len=max_len)
    src = torch.tensor([ids], device=device)
    src_mask = model.make_src_mask(src)
    enc = model.encode(src, src_mask)
    ys = torch.full((1, 1), BOS_IDX, dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys)
        out = model.decode(ys, enc, src_mask, tgt_mask)
        nxt = out[:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], 1)
        if nxt.item() == EOS_IDX:
            break
    ids = [i for i in ys[0].tolist() if i not in (PAD_IDX, BOS_IDX, EOS_IDX)]
    return tgt_tok.decode(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="./weights/latest.pt")
    p.add_argument("sentences", nargs="+")
    args = p.parse_args()

    model, src_tok, tgt_tok, device, max_len = load_model(args.ckpt)
    for s in args.sentences:
        print(f"EN: {s}")
        print(f"UR: {translate(model, src_tok, tgt_tok, device, s, max_len)}")
        print()


if __name__ == "__main__":
    main()
