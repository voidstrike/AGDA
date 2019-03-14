from torch import nn
from torch.nn import init


#  Classifier -- dimension of hidden space equals 100
class LinearClf100(nn.Module):
    def __init__(self):
        super(LinearClf100, self).__init__()

        # Create Sequential Model
        self.model = nn.Sequential(
            nn.Linear(100, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 10)
        )

        # Initialize the weight via Xavier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

    def forward(self, x):
        return self.model(x)


#  Classifier -- dimension of hidden space equals 100
class LinearClf400(nn.Module):
    def __init__(self):
        super(LinearClf400, self).__init__()

        # Create Sequential Model
        self.model = nn.Sequential(
            nn.Linear(400, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 84),
            nn.LeakyReLU(),
            nn.Linear(84, 10),
            #nn.Softmax(dim=1)
        )

        # Initialize the weight via Xavier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

    def forward(self, x):
        return self.model(x)


#  GAN - Discriminator - dimension of hidden space equals 100
class Discriminator100(nn.Module):
    def __init__(self):
        super(Discriminator100, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


#  GAN - Discriminator - dimension of hidden space equals 400
class Discriminator400(nn.Module):
    def __init__(self):
        super(Discriminator400, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(400, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
