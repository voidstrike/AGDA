from torch import nn


# Convolution Auto Encoder
# Hardcoded model, modify while required
class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        # Encoder Network
        # Output Shape = [(Original_Shape - Kernel + 2 * Padding) / Stride] + 1
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=3, padding=1),  # (b, 6, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(6, 16, 3, stride=2, padding=1),  # (b, 16, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 16, 2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 6, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        code = self.encoder(x)
        rec = self.decoder(code)
        code = code.flatten(start_dim=1)
        rec = rec.view(-1, 28 * 28)
        return code, rec