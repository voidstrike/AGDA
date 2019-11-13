import torch.nn as nn
import torch.nn.functional as func
import torch
from torch.autograd import Variable
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, features, drop_out=.0):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(features, features, 3, padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.InstanceNorm2d(features),
        ]

        if drop_out != .0:
            conv_block.append(nn.Dropout(p=drop_out))

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Encoder(nn.Module):
    # Structure in the paper : 3 Conv layers + 4 residual blocks
    def __init__(self, in_channels=3, dim=64, n_downsample=2, shared_block=None, drop_out=.5):
        super(Encoder, self).__init__()

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim, drop_out=drop_out)]

        self.model_blocks = nn.Sequential(*layers)
        self.shared_block = shared_block

    def reparameterization(self, mu):
        # Reduced VAE -> The outputs are multivariate Gaussian distribution with mean = mu & std = 1
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return z + mu

    def forward(self, x):
        x = self.model_blocks(x)
        mu = self.shared_block(x)
        z = self.reparameterization(mu)
        return mu, z


class Decoder(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None, drop_out=.5):
        super(Decoder, self).__init__()

        self.shared_block = shared_block

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim, drop_out=drop_out)]

        # Up-sampling
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.Conv2d(dim, out_channels, 1, stride=1), nn.Tanh()]

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.shared_block(x)
        x = self.model_blocks(x)
        return x


class DisBlock(nn.Module):
    def __init__(self, in_filters, out_filters, normalize=True):
        super(DisBlock, self).__init__()
        # Basic Discriminator Block -> kernel_size=4 & stride=2 => reduce the height & width by half
        layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 3, width // 2 ** 3)

        self.model = nn.Sequential(
            DisBlock(channels, 64, normalize=False),
            DisBlock(64, 128),
            DisBlock(128, 256),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, img):
        return self.model(img)


class AttnDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(AttnDiscriminator, self).__init__()
        channels, height, width = input_shape
        self.mask_shape = height
        self.output_shape = (1, height // 2 ** 3, width // 2 ** 3)

        self.model = nn.Sequential(
            DisBlock(channels, 64, normalize=False),
            DisBlock(64, 128),
            DisBlock(128, 256)
        )

        self.fin = nn.Conv2d(256, 1, 1)

    def forward(self, img):
        inter = self.model(img)
        expand_mask = self._mask_rescale(inter)
        fin = self.fin(inter)
        return fin, expand_mask

    def _mask_rescale(self, mask_tensor):
        mask_tensor = torch.mean(abs(mask_tensor), dim=1).unsqueeze(1)
        t_max, t_min = torch.max(mask_tensor), torch.min(mask_tensor)
        mask_tensor = (mask_tensor - t_min) / (t_max - t_min)
        return func.interpolate(mask_tensor, (self.mask_shape, self.mask_shape), mode='bilinear')

