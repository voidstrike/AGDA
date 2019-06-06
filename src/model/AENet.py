import torch

from torch import nn
from model.ConvNet import ConvBlock, DeConvBlock


# General Auto Encoder Architecture, contains one encoder & one decoder
class GeneralAE(nn.Module):
    def __init__(self, img_static, encoder=None, decoder=None, flat=True):
        super(GeneralAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.img_channel = img_static[0]
        self.img_height = img_static[1]
        self.img_width = img_static[2]
        self.flat = flat

    def forward(self, x):
        code = self.encoder(x)
        rec = self.decoder(code)

        if self.flat:
            code = code.flatten(start_dim=1)
        rec = rec.view(-1, self.img_channel * self.img_height * self.img_width)

        return code, rec


# General Variational AE Architecture, one feature extractor + two encoder and one decoder
class GeneralVAE(nn.Module):
    def __init__(self, img_static, feature_extractor=None, mean_net=None, std_net=None,
                 decoder=None, flat=True):
        super(GeneralVAE, self).__init__()
        self.feature_extractor = feature_extractor
        self.mean_net, self.std_net = mean_net, std_net
        self.decoder = decoder
        self.img_channel = img_static[0]
        self.img_height = img_static[1]
        self.img_width = img_static[2]
        self.flat = flat

    def forward(self, x):
        feature = self.feature_extractor(x)

        mu = self.mean_net(feature)
        logvar = self.std_net(feature)

        code = self.reparameterize(mu, logvar)
        rec = self.decoder(code)

        if self.flat:
            code = code.flatten(start_dim=1)
        rec = rec.view(-1, self.img_channel * self.img_height * self.img_width)

        return code, rec

    def reparameterize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std