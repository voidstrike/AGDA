from torch import nn
import torch
import torch.nn.functional as F


# Standard Encoder of MUNIT architecture
class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2, style_dim=8):
        super(Encoder, self).__init__()
        self.content_encoder = ContentEncoder(in_channels, dim, n_residual, n_downsample)
        self.style_encoder = StyleEncoder(in_channels, dim, n_downsample, style_dim)

    def forward(self, x):
        content_code = self.content_encoder(x)
        style_code = self.style_encoder(x)
        return content_code, style_code


class ContentEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=4, n_downsample=2):
        super(ContentEncoder, self).__init__()

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim * 2, 4, stride=2),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="in")]

        self.model = nn.Sequential(*layers)
        self.output_dim = dim  # 64

    def forward(self, x):
        return self.model(x)


class StyleEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2, style_dim=8):
        super(StyleEncoder, self).__init__()

        # Initial conv block
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, dim, 7), nn.ReLU(inplace=True)]

        # Downsampling
        for _ in range(2):
            layers += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim * 2, 4, stride=2), nn.ReLU(inplace=True)]
            dim *= 2

        # Downsampling -- Doesn't increase the number of filters
        for _ in range(n_downsample - 2):
            layers += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 4, stride=2), nn.ReLU(inplace=True)]

        # Average pool and output layer
        layers += [nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*layers)
        self.output_dim = dim  # (n, 8, 1, 1)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_residual=4, n_upsample=2, style_dim=8):
        super(Decoder, self).__init__()

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="adain")]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.ReflectionPad2d(2),
                nn.Conv2d(dim, dim // 2, 5, stride=1),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*layers)

        # Initiate mlp (predicts AdaIN parameters)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(style_dim, num_adain_params)

    def get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # Extract mean and std predictions
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def forward(self, content_code, style_code):
        # Update AdaIN parameters by MLP prediction based off style code
        self.assign_adain_params(self.mlp(style_code))
        img = self.model(content_code)
        return img


class DisBlock(nn.Module):
    def __init__(self, in_filters, out_filters, normalize=False):
        super(DisBlock, self).__init__()
        # Basic Discriminator Block -> kernel_size=4 & stride=2 => reduce the height & width by half
        layers = [nn.ReflectionPad2d(1), nn.Conv2d(in_filters, out_filters, 4, stride=2)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    DisBlock(in_channels, 64),
                    DisBlock(64, 128),
                    DisBlock(128, 256),
                    DisBlock(256, 512),
                    nn.Conv2d(512, 1, 1, stride=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, features, norm="in"):
        super(ResidualBlock, self).__init__()

        # initialize normalization
        if norm == 'bn':
            norm_layer = nn.BatchNorm2d
        elif norm == 'in':
            norm_layer = nn.InstanceNorm2d
        elif norm == 'ln':
            norm_layer = LayerNorm
        elif norm == 'adain':
            norm_layer = AdaptiveInstanceNorm2d
        elif norm == 'none' or norm == 'sn':
            norm_layer = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
        )

    def forward(self, x):
        return x + self.block(x)


class AdaptiveInstanceNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))