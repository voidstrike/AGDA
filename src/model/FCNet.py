from torch import nn


# Elemental Structure of a Fully Connected Network
class FCBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, norm=None, init=None):
        super(FCBlock, self).__init__()
        modules = [nn.Linear(in_dim, out_dim)]

        if norm is not None:
            if norm == 'batch':
                modules.append(nn.BatchNorm1d(out_dim))
            elif norm == 'instance':
                modules.append(nn.InstanceNorm1d(out_dim))
            else:
                raise Exception('Unsupported Normalization Method Selected')

        if activation is not None:
            if activation == 'relu':
                modules.append(nn.ReLU())
            elif activation == 'leaky':
                modules.append(nn.LeakyReLU())
            else:
                raise Exception('Unsupported Activation Func Selected')

        self.model = nn.Sequential(*modules)

        if init is not None:
            self.weight_init(init)

    def weight_init(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                else:
                    raise Exception('Unsupported Weight Initialization Method')

    def forward(self, x):
        return self.model(x)


# General FC Classifier/Discriminator
# Contains a sequence of FCBlocks and a N->out_dim Linear layer in the end
class MLPNet(nn.Module):
    def __init__(self, layers, activation='relu', norm=None, out_dim=1):
        super(MLPNet, self).__init__()
        modules = []

        for idx in range(1, len(layers)):
            modules.append(FCBlock(layers[idx], layers[idx-1], activation, norm, init=None))

        modules.append(nn.Linear(layers[-1], out_dim))

        self.model = nn.Sequential(*modules)

    def weight_init(self, method='xavier'):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                else:
                    raise Exception('Unsupported Weight Initialization Method')

    def forward(self, x):
        return self.model(x)
