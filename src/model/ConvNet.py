from torch import nn


# Basic Convolution Block
# Conv2d - Norm - Activation - Pooling
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, padding=0, activation=None, norm=None, pooling=None, pooling_stride=None,
                 init=None):
        super(ConvBlock, self).__init__()
        modules = [nn.Conv2d(in_c, out_c, stride, padding)]

        if norm is not None:
            if norm == 'batch':
                modules.append(nn.BatchNorm2d(out_c))
            elif norm == 'instance':
                modules.append(nn.InstanceNorm2d(out_c))
            else:
                raise Exception('Unsupported Normalization Method Selected')

        if activation is not None:
            if activation == 'relu':
                modules.append(nn.ReLU())
            elif activation == 'leaky':
                modules.append(nn.LeakyReLU())
            else:
                raise Exception('Unsupported Activation Func Selected')

        if pooling is not None:
            if pooling == 'avg':
                modules.append(nn.AvgPool2d(pooling_stride))
            elif pooling == 'max':
                modules.append(nn.MaxPool2d(pooling_stride))
            else:
                raise Exception('Unsupported Pooling Method Selected')

        self.model = nn.Sequential(*modules)

        if init is not None:
            self.weight_init(init)

    def weight_init(self, method):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                else:
                    raise Exception('Unsupported Weight Initialization Method')

    def forward(self, x):
        return self.model(x)


# Basic Deconvolution Block
# Conv2dTranspose - Norm - Activation - Pooling
class DeConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, padding=0, activation=None, norm=None, pooling=None, pooling_stride=None,
                 init=None):
        super(DeConvBlock, self).__init__()
        modules = [nn.ConvTranspose2d(in_c, out_c, stride, padding)]

        if norm is not None:
            if norm == 'batch':
                modules.append(nn.BatchNorm2d(out_c))
            elif norm == 'instance':
                modules.append(nn.InstanceNorm2d(out_c))
            else:
                raise Exception('Unsupported Normalization Method Selected')

        if activation is not None:
            if activation == 'relu':
                modules.append(nn.ReLU())
            elif activation == 'leaky':
                modules.append(nn.LeakyReLU())
            else:
                raise Exception('Unsupported Activation Func Selected')

        if pooling is not None:
            if pooling == 'avg':
                modules.append(nn.AvgPool2d(pooling_stride))
            elif pooling == 'max':
                modules.append(nn.MaxPool2d(pooling_stride))
            else:
                raise Exception('Unsupported Pooling Method Selected')

        self.model = nn.Sequential(*modules)

        if init is not None:
            self.weight_init(init)

    def weight_init(self, method):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                if method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                else:
                    raise Exception('Unsupported Weight Initialization Method')

    def forward(self, x):
        return self.model(x)
