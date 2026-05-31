"""
CycleGAN model definitions, faithful to Zhu et al. 2017.

- Generator: ResNet-based (9 residual blocks for 256x256 input, as in paper).
- Discriminator: 70x70 PatchGAN.
- Weights init: Normal(0, 0.02) as in the paper.

Two directions:
    G_AB : A -> B  (real face  -> sketch)
    G_BA : B -> A  (sketch     -> real face)
and discriminators D_A (real/fake faces) and D_B (real/fake sketches).
"""
import torch
import torch.nn as nn


def weights_init_normal(m):
    """Normal(0, 0.02) init used in the CycleGAN paper."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("InstanceNorm2d") != -1 and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    """ResNet block with reflection padding and InstanceNorm (paper spec)."""
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    """ResNet generator: c7s1-64, d128, d256, R256 x n_residual, u128, u64, c7s1-3."""
    def __init__(self, input_nc=3, output_nc=3, num_residual_blocks=9):
        super().__init__()
        channels = 64

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, channels, 7),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = channels
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(channels, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """70x70 PatchGAN used in CycleGAN."""
    def __init__(self, input_nc=3):
        super().__init__()

        def block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_nc, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # quick shape sanity check
    g = GeneratorResNet().eval()
    d = Discriminator().eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = g(x)
        p = d(y)
    print("G out:", y.shape)   # (1,3,256,256)
    print("D out:", p.shape)   # (1,1,30,30) PatchGAN
