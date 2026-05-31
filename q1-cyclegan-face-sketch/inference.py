"""
Inference helpers for CycleGAN.

Public API:
    load_generators(weights_path, device) -> (G_AB, G_BA)
    detect_sketch(pil_img) -> bool           # heuristic: sketches are ~grayscale
    translate(pil_img, G_AB, G_BA, device) -> (output_pil, mode)
        mode in {"face->sketch", "sketch->face"}
"""
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from cyclegan_model import GeneratorResNet


IMG_SIZE = 256


def _pre(image_size=IMG_SIZE):
    return T.Compose([
        T.Resize((image_size, image_size), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


def _tensor_to_pil(t):
    t = (t.clamp(-1, 1) + 1) / 2
    t = (t.detach().cpu().numpy() * 255).astype(np.uint8)
    # (3, H, W) -> (H, W, 3)
    t = np.transpose(t, (1, 2, 0))
    return Image.fromarray(t)


def load_generators(weights_path, device=None):
    """Load G_AB (face->sketch) and G_BA (sketch->face)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_AB = GeneratorResNet(3, 3, 9).to(device).eval()
    G_BA = GeneratorResNet(3, 3, 9).to(device).eval()
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}. "
            "Train first (see COLAB_INSTRUCTIONS.md) or drop weights into Q1_CycleGAN/weights/."
        )
    ck = torch.load(weights_path, map_location=device)
    # Accept both generators-only files and full checkpoint files
    state_G_AB = ck.get("G_AB") or ck["state_dict"]["G_AB"]
    state_G_BA = ck.get("G_BA") or ck["state_dict"]["G_BA"]
    G_AB.load_state_dict(state_G_AB)
    G_BA.load_state_dict(state_G_BA)
    return G_AB, G_BA, device


def detect_sketch(pil_img, saturation_threshold=0.12, sat_pixel_pct=0.08):
    """Robust heuristic: a pencil sketch contains almost no saturated pixels
    (it's black strokes on white). A photograph--even with a plain white
    background--has many saturated pixels concentrated on the subject.

    We compute HSV-style saturation per pixel and count the fraction whose
    saturation exceeds `saturation_threshold`. If fewer than
    `sat_pixel_pct` of pixels are saturated, the image is treated as a sketch.

    The previous heuristic (mean colour deviation) misclassified studio
    photos with white backgrounds because the background dilutes the global
    colour statistic.
    """
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    saturation = (mx - mn) / np.maximum(mx, 1e-6)
    return float((saturation > saturation_threshold).mean()) < sat_pixel_pct


@torch.no_grad()
def translate(pil_img, G_AB, G_BA, device, force=None):
    """force in {None, 'face2sketch', 'sketch2face'}. None = auto-detect."""
    mode = force
    if mode is None:
        mode = "sketch2face" if detect_sketch(pil_img) else "face2sketch"

    x = _pre()(pil_img.convert("RGB")).unsqueeze(0).to(device)
    if mode == "face2sketch":
        y = G_AB(x)[0]
        label = "face->sketch"
    else:
        y = G_BA(x)[0]
        label = "sketch->face"
    return _tensor_to_pil(y), label


if __name__ == "__main__":
    # Tiny CPU smoke test: random tensor through freshly-inited generators.
    G_AB = GeneratorResNet().eval()
    G_BA = GeneratorResNet().eval()
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    out, label = translate(img, G_AB, G_BA, torch.device("cpu"), force="face2sketch")
    print("Smoke test OK:", label, out.size)
