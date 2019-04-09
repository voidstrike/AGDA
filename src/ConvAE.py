from torch import nn
from torch.nn import init
from torch.autograd import Variable
from copy import deepcopy
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
    def __init__(self, exl_flag=False):
        super(ExLeNetAE28, self).__init__()

        # Encoder Network
        self.encoder_cnn = nn.Sequential(
            # DynamicGNoise(28, std=0.05),
            nn.Conv2d(1, 20, 5, stride=1),  # (b, 20, 24, 24)
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (b, 20, 12, 12)
            # DynamicGNoise(12, std=0.05),
            nn.Conv2d(20, 50, 5, stride=1),  # (b, 50, 8, 8)
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # (b, 16, 4, 4)
        )

        # Extra layer to modify extracted feature
        self.extra_flag = exl_flag
        self.extra_layer = nn.Sequential(
            nn.Linear(800, 500),
            nn.ReLU()
        )

        # Decoder Network
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(50, 20, 5, stride=3, padding=1),  # (b, 20, 12, 12)
            nn.ReLU(True),
            nn.ConvTranspose2d(20, 1, 6, stride=2, padding=0),  # (b, 1, 28, 28)
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

        if self.extra_flag:
            code = self.extra_layer(code)

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
            nn.MaxPool2d(2, stride=2),  # (b, 6, 14, 14)
            # DynamicGNoise(14, std=0.05),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # (b, 16, 5, 5)
        )

        # Decoder Network
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(16, 6, 3, stride=3, padding=0),  # (b, 6, 15, 15)
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


# TODO
class ExAlexNet(nn.Module):
    def __init__(self, feature_extractor=None):
        super(ExAlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(6)

        if feature_extractor is not None:
            tmp_count = 1
            for module in feature_extractor:
                if isinstance(module, nn.Conv2d):
                    if tmp_count == 1:
                        self.conv1 = deepcopy(module)
                        self.conv1.requires_grad = False
                    elif tmp_count == 2:
                        self.conv2 = deepcopy(module)
                        self.conv2.requires_grad = False
                    elif tmp_count == 3:
                        self.conv3 = deepcopy(module)
                        self.conv3.requires_grad = False
                    elif tmp_count == 4:
                        self.conv4 = deepcopy(module)
                        self.conv4.requires_grad = False
                    elif tmp_count == 5:
                        self.conv5 = deepcopy(module)
                    tmp_count += 1

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 384, kernel_size=5, stride=1),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        intermediate_img = x

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)
        final_feature = self.avgpool(x)
        rec_img = self.decoder(final_feature)

        final_feature = final_feature.flatten(start_dim=1)
        intermediate_img = intermediate_img.view(-1, 384 * 13 * 13)
        rec_img = rec_img.view(-1, 384 * 13 * 13)

        return intermediate_img, final_feature, rec_img

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

