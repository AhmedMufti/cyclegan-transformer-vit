"""
Transformer encoder-decoder (Vaswani et al. 2017) for English -> Urdu MT.

Everything is implemented from scratch in PyTorch (no nn.Transformer shortcut)
so the architecture can be inspected component-by-component during the demo.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (eq. from the paper)."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.shape
        Lk = k.size(1)
        Q = self.q(q).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k(k).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v(v).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask: broadcastable to (B, h, Lq, Lk); True/1 = keep
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out(ctx)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.ln1(x + self.drop(self.self_attn(x, x, x, src_mask)))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model); self.ln3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc, src_mask, tgt_mask):
        x = self.ln1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.ln2(x + self.drop(self.cross_attn(x, enc, enc, src_mask)))
        x = self.ln3(x + self.drop(self.ff(x)))
        return x


class Transformer(nn.Module):
    """Standard encoder-decoder Transformer."""
    def __init__(self, src_vocab, tgt_vocab, d_model=256, n_heads=8,
                 n_enc=4, n_dec=4, d_ff=1024, dropout=0.1, max_len=256, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc)])
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec)])
        self.proj = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model

    def make_src_mask(self, src):
        # (B, 1, 1, S) -> broadcasts to (B, h, L, S)
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        B, T = tgt.shape
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)          # (B,1,1,T)
        causal = torch.tril(torch.ones(T, T, device=tgt.device, dtype=torch.bool))
        return pad_mask & causal.unsqueeze(0).unsqueeze(0)                  # (B,1,T,T)

    def encode(self, src, src_mask):
        x = self.drop(self.pos(self.src_emb(src) * math.sqrt(self.d_model)))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc, src_mask, tgt_mask):
        x = self.drop(self.pos(self.tgt_emb(tgt) * math.sqrt(self.d_model)))
        for layer in self.dec_layers:
            x = layer(x, enc, src_mask, tgt_mask)
        return self.proj(x)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc = self.encode(src, src_mask)
        return self.decode(tgt, enc, src_mask, tgt_mask)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = Transformer(1000, 1000)
    src = torch.randint(1, 1000, (2, 10))
    tgt = torch.randint(1, 1000, (2, 12))
    out = m(src, tgt)
    print("out:", out.shape, "params:", count_params(m))
