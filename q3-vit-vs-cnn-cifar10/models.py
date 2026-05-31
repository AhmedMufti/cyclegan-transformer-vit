"""
Model definitions for Q3: a custom CNN and a from-scratch Vision Transformer
for CIFAR-10 classification.

Both operate on 32x32 RGB images.
"""
import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# CNN baseline (ResNet-ish: conv + BN + ReLU + residual)
# --------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.skip = (nn.Identity() if (stride == 1 and in_c == out_c)
                     else nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                                        nn.BatchNorm2d(out_c)))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.skip(x))


class SimpleCNN(nn.Module):
    """ResNet-20-ish for CIFAR-10 (~270k params)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


# --------------------------------------------------------------------------
# Vision Transformer (from scratch, following the ViT paper)
# --------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, d_model=192):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        # Conv with stride=patch_size is equivalent to flatten-then-linear projection
        self.proj = nn.Conv2d(in_ch, d_model, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)              # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)   # (B, N, D)
        return x


class ViTBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    """Compact ViT: patch 4x4, d_model=192, depth=6, heads=6."""
    def __init__(self, img_size=32, patch_size=4, num_classes=10,
                 d_model=192, depth=6, n_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, d_model)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            ViTBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], 1)
        x = self.drop(x + self.pos_embed)
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        return self.head(x[:, 0])


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    cnn = SimpleCNN(); vit = ViT()
    x = torch.randn(2, 3, 32, 32)
    print("CNN out:", cnn(x).shape, "params:", count_params(cnn))
    print("ViT out:", vit(x).shape, "params:", count_params(vit))
