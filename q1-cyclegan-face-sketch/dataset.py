"""
Paired/unpaired dataloader for the Person Face Sketches dataset.

Expected structure after unzipping the Kaggle archive:
    root/
        photos/      <- real face images
        sketches/    <- pencil sketches

CycleGAN does not require paired data, but this dataset happens to be paired.
We still draw from each domain independently (unpaired style) to stay faithful
to the CycleGAN formulation.
"""
import os
import random
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files += glob(os.path.join(folder, f"*{ext}"))
        files += glob(os.path.join(folder, f"*{ext.upper()}"))
    return sorted(files)


def make_transform(image_size=256, train=True):
    """Standard CycleGAN preprocessing: resize to 286, random crop 256, flip."""
    if train:
        return T.Compose([
            T.Resize(int(image_size * 1.12), T.InterpolationMode.BICUBIC),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return T.Compose([
        T.Resize((image_size, image_size), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class FaceSketchDataset(Dataset):
    """Unpaired sampling: index into A is deterministic, B is random."""
    def __init__(self, root, image_size=256, train=True,
                 photos_dir="photos", sketches_dir="sketches"):
        self.files_A = _list_images(os.path.join(root, photos_dir))
        self.files_B = _list_images(os.path.join(root, sketches_dir))
        if not self.files_A or not self.files_B:
            raise RuntimeError(
                f"No images found. Expected {root}/{photos_dir} and "
                f"{root}/{sketches_dir}"
            )
        self.transform = make_transform(image_size, train)
        self.train = train

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def _load(self, path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        a_path = self.files_A[idx % len(self.files_A)]
        if self.train:
            b_path = random.choice(self.files_B)
        else:
            b_path = self.files_B[idx % len(self.files_B)]
        a = self.transform(self._load(a_path))
        b = self.transform(self._load(b_path))
        return {"A": a, "B": b}
