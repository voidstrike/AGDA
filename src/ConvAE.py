from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch


# Convolution Auto Encoder
class LeNetAE28(nn.Module):
    def __init__(self):
        super(LeNetAE28, self).__init__()

        # Encoder Network
        self.encoder_cnn = nn.Sequential(
            DynamicGNoise(28, std=0.05),
            nn.Conv2d(1, 6, 5, stride=1, padding=2),  # (b, 6, 28, 28)
            nn.LeakyReLU(),
            nn.AvgPool2d(2, stride=2),  # (b, 16, 14, 14)
            DynamicGNoise(14, std=0.05),
            nn.Conv2d(6, 16, 5, stride=1),  # (b, 16, 10, 10)
            nn.LeakyReLU(),
            nn.AvgPool2d(2, stride=2)  # (b, 16, 5, 5)
        )

        # Decoder Network
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 2, stride=2),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 6, 5, stride=1),  # (b, 6, 14, 14)
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, 4, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                init.xavier_uniform_(module.weight)

    def forward(self, x):
        code = self.encoder_cnn(x)
        rec = self.decoder_cnn(code)

        # Modified the shape of each return tensor
        code = code.flatten(start_dim=1)
        rec = rec.view(-1, 28 * 28)

        return code, rec

    # Auxiliary function that controls how many layers are not trainable
    def setPartialTrainable(self, num_layer=0):
        if num_layer != 0:
            ct = 0
            for eachLayer in self.encoder_cnn:
                if isinstance(eachLayer, torch.nn.Conv2d):
                    if ct < num_layer:
                        eachLayer.requires_grad = False
                        ct += 1
                    else:
                        init.xavier_uniform_(eachLayer.weight)


# Convolution Auto Encoder
class ExLeNetAE28(nn.Module):
    def __init__(self):
        super(ExLeNetAE28, self).__init__()

        # Encoder Network
        self.encoder_cnn = nn.Sequential(
            # DynamicGNoise(28, std=0.05),
            nn.Conv2d(1, 20, 5, stride=1),  # (b, 20, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (b, 20, 12, 12)
            # DynamicGNoise(12, std=0.05),
            nn.Conv2d(20, 50, 5, stride=1),  # (b, 50, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # (b, 16, 4, 4)
        )

        # Decoder Network
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(50, 20, 5, stride=3, padding=1),  # (b, 20, 12, 12)
            nn.ReLU(True),
            nn.ConvTranspose2d(20, 1, 4, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.ReLU(True),
            nn.Tanh()
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                init.xavier_uniform_(module.weight)

    def forward(self, x):
        code = self.encoder_cnn(x)
        rec = self.decoder_cnn(code)

        # Modified the shape of each return tensor
        code = code.flatten(start_dim=1)
        rec = rec.view(-1, 28 * 28)

        return code, rec

    # Auxiliary function that controls how many layers are not trainable
    def setPartialTrainable(self, num_layer=0):
        if num_layer != 0:
            ct = 0
            for eachLayer in self.encoder_cnn:
                if isinstance(eachLayer, torch.nn.Conv2d):
                    if ct < num_layer:
                        eachLayer.requires_grad = False
                        ct += 1
                    else:
                        init.xavier_uniform_(eachLayer.weight)


class LeNetAE32(nn.Module):
    def __init__(self):
        super(LeNetAE32, self).__init__()

        # Encoder Network
        self.encoder_cnn = nn.Sequential(
            # DynamicGNoise(32, std=0.05),
            nn.Conv2d(1, 6, 5, stride=1, padding=0),  # (b, 6, 28, 28)
            nn.ReLU(True),
            nn.AvgPool2d(2, stride=2),  # (b, 6, 14, 14)
            # DynamicGNoise(14, std=0.05),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.AvgPool2d(2, stride=2)  # (b, 16, 5, 5)
        )

        # Decoder Network
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 2, stride=2),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 6, 5, stride=1, padding=0),  # (b, 6, 14, 14)
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, 4, stride=2, padding=0),  # (b, 1, 32, 32)
            nn.Tanh()
        )

        # Default Initialize Process -- Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                init.xavier_uniform_(module.weight)

    def forward(self, x):
        code = self.encoder_cnn(x)
        rec = self.decoder_cnn(code)

        # Modified the sharp of each return tensor
        code = code.flatten(start_dim=1)
        rec = rec.view(-1, 32 * 32)
        return code, rec

    # Auxiliary function that controls how many layers are not trainable
    def setPartialTrainable(self, num_layer=0):
        if num_layer != 0:
            ct = 0
            for eachLayer in self.encoder_cnn:
                if isinstance(eachLayer, torch.nn.Conv2d):
                    if ct < num_layer:
                        eachLayer.requires_grad = False
                        ct += 1
                    else:
                        init.xavier_uniform_(eachLayer.weight)


class DynamicGNoise(nn.Module):
    def __init__(self, shape, std=0.05):
        super(DynamicGNoise, self).__init__()

        self.noise = Variable(torch.zeros(shape, shape))
        if torch.cuda.is_available():
            self.noise = self.noise.cuda()

        self.std = std

    def forward(self, x):
        if not self.training:
            return x

        self.noise.data.normal_(0, std=self.std)
        return x + self.noise.expand_as(x)
