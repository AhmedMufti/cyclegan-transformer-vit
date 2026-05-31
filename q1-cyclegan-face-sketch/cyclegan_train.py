"""
CycleGAN training loop.

Losses (Zhu et al. 2017):
    L_GAN(G_AB, D_B)       : LSGAN adversarial for A->B
    L_GAN(G_BA, D_A)       : LSGAN adversarial for B->A
    L_cycle                : ||G_BA(G_AB(A)) - A||_1 + ||G_AB(G_BA(B)) - B||_1
    L_identity             : ||G_BA(A) - A||_1 + ||G_AB(B) - B||_1  (half weight)

Total G loss: L_GAN + lambda_cycle * L_cycle + lambda_id * L_identity
Default: lambda_cycle = 10, lambda_id = 5.

Resume support: if --resume is passed, loads the latest weights/epoch file.
Weights are saved after every epoch to weights/.
"""
import argparse
import itertools
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cyclegan_model import (Discriminator, GeneratorResNet,
                            weights_init_normal)
from dataset import FaceSketchDataset


# ------------------------------------------------------------------
# Replay buffer (keeps last 50 generated images per the paper)
# ------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        out = []
        for el in data:
            el = el.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(el)
                out.append(el)
            else:
                if torch.rand(1).item() < 0.5:
                    i = torch.randint(0, self.max_size, (1,)).item()
                    out.append(self.data[i].clone())
                    self.data[i] = el
                else:
                    out.append(el)
        return torch.cat(out)


def lambda_lr(epoch, n_epochs, decay_start):
    """Linear LR decay after decay_start."""
    if epoch < decay_start:
        return 1.0
    return max(0.0, 1.0 - (epoch - decay_start) / float(n_epochs - decay_start))


def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    # Models
    G_AB = GeneratorResNet(3, 3, num_residual_blocks=9).to(device)
    G_BA = GeneratorResNet(3, 3, num_residual_blocks=9).to(device)
    D_A = Discriminator(3).to(device)
    D_B = Discriminator(3).to(device)
    for m in (G_AB, G_BA, D_A, D_B):
        m.apply(weights_init_normal)

    # Losses
    criterion_gan = torch.nn.MSELoss()       # LSGAN
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers
    optim_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
    )
    optim_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # LR schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optim_G, lr_lambda=lambda e: lambda_lr(e, args.n_epochs, args.decay_epoch))
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optim_D_A, lr_lambda=lambda e: lambda_lr(e, args.n_epochs, args.decay_epoch))
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optim_D_B, lr_lambda=lambda e: lambda_lr(e, args.n_epochs, args.decay_epoch))

    # Resume
    start_epoch = 0
    ckpt_path = Path(args.weights_dir) / "latest.pt"
    if args.resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ck = load_checkpoint(ckpt_path, device)
        G_AB.load_state_dict(ck["G_AB"]); G_BA.load_state_dict(ck["G_BA"])
        D_A.load_state_dict(ck["D_A"]);   D_B.load_state_dict(ck["D_B"])
        optim_G.load_state_dict(ck["optim_G"])
        optim_D_A.load_state_dict(ck["optim_D_A"])
        optim_D_B.load_state_dict(ck["optim_D_B"])
        start_epoch = ck["epoch"] + 1
        for _ in range(start_epoch):
            lr_scheduler_G.step(); lr_scheduler_D_A.step(); lr_scheduler_D_B.step()

    # Data
    ds = FaceSketchDataset(args.data_root, image_size=args.image_size, train=True,
                           photos_dir=args.photos_dir, sketches_dir=args.sketches_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    for epoch in range(start_epoch, args.n_epochs):
        t0 = time.time()
        for i, batch in enumerate(loader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # ----------- Train Generators -----------
            G_AB.train(); G_BA.train()
            optim_G.zero_grad()

            # Identity
            id_A = G_BA(real_A); loss_id_A = criterion_identity(id_A, real_A)
            id_B = G_AB(real_B); loss_id_B = criterion_identity(id_B, real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN
            fake_B = G_AB(real_A)
            pred_fake_B = D_B(fake_B)
            valid = torch.ones_like(pred_fake_B, device=device)
            loss_gan_AB = criterion_gan(pred_fake_B, valid)

            fake_A = G_BA(real_B)
            pred_fake_A = D_A(fake_A)
            loss_gan_BA = criterion_gan(pred_fake_A, valid)

            loss_gan = (loss_gan_AB + loss_gan_BA) / 2

            # Cycle
            recov_A = G_BA(fake_B); loss_cyc_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A); loss_cyc_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cyc_A + loss_cyc_B) / 2

            loss_G = loss_gan + args.lambda_cycle * loss_cycle + args.lambda_id * loss_identity
            loss_G.backward()
            optim_G.step()

            # ----------- Train D_A -----------
            optim_D_A.zero_grad()
            pred_real = D_A(real_A)
            loss_real = criterion_gan(pred_real, torch.ones_like(pred_real, device=device))
            fake_A_ = fake_A_buffer.push_and_pop(fake_A.detach())
            pred_fake = D_A(fake_A_)
            loss_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake, device=device))
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optim_D_A.step()

            # ----------- Train D_B -----------
            optim_D_B.zero_grad()
            pred_real = D_B(real_B)
            loss_real = criterion_gan(pred_real, torch.ones_like(pred_real, device=device))
            fake_B_ = fake_B_buffer.push_and_pop(fake_B.detach())
            pred_fake = D_B(fake_B_)
            loss_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake, device=device))
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optim_D_B.step()

            if i % args.log_every == 0:
                print(f"[ep {epoch}/{args.n_epochs}] [{i}/{len(loader)}] "
                      f"G {loss_G.item():.3f} | cyc {loss_cycle.item():.3f} | "
                      f"id {loss_identity.item():.3f} | "
                      f"D_A {loss_D_A.item():.3f} | D_B {loss_D_B.item():.3f}")

        # Epoch done
        lr_scheduler_G.step(); lr_scheduler_D_A.step(); lr_scheduler_D_B.step()

        # Samples
        with torch.no_grad():
            G_AB.eval(); G_BA.eval()
            sample_real_A = real_A[:4]
            sample_real_B = real_B[:4]
            sample_fake_B = G_AB(sample_real_A)
            sample_fake_A = G_BA(sample_real_B)
            grid = torch.cat([sample_real_A, sample_fake_B, sample_real_B, sample_fake_A], 0)
            save_image((grid + 1) / 2, f"{args.samples_dir}/epoch_{epoch:03d}.png", nrow=4)

        # Save every epoch (assignment requirement)
        ckpt = {
            "epoch": epoch,
            "G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
            "D_A": D_A.state_dict(),   "D_B": D_B.state_dict(),
            "optim_G": optim_G.state_dict(),
            "optim_D_A": optim_D_A.state_dict(),
            "optim_D_B": optim_D_B.state_dict(),
        }
        save_checkpoint(ckpt, Path(args.weights_dir) / "latest.pt")
        save_checkpoint(ckpt, Path(args.weights_dir) / f"epoch_{epoch:03d}.pt")
        # Light-weight generator-only weights for inference
        torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict()},
                   Path(args.weights_dir) / "generators.pt")

        print(f"Epoch {epoch} done in {time.time() - t0:.1f}s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--photos_dir", type=str, default="photos")
    p.add_argument("--sketches_dir", type=str, default="sketches")
    p.add_argument("--weights_dir", type=str, default="./weights")
    p.add_argument("--samples_dir", type=str, default="./samples")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--decay_epoch", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_cycle", type=float, default=10.0)
    p.add_argument("--lambda_id", type=float, default=5.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
