from torch import nn

from model.ConvNet import ConvBlock
from model.FCNet import MLPNet


# Well known LeNet5, take input shape (Batch * 3 * 32 * 32)
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 6, 5, 1, activation='relu', pooling='max', pooling_stride=2),
            ConvBlock(6, 16, 5, 1, activation='relu', pooling='max', pooling_stride=2),
        )

        self.clf_model = MLPNet([400, 400, 200], out_dim=10)

    def forward(self, x):
        f = self.feature_extractor(x)
        return self.clf_model(f.view(-1, 400))
