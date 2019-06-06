from torch import nn
from torch.nn import init


# General version of classifier -- Configure the network structure in params.py
class GClassifier(nn.Module):
    def __init__(self, layer_list, relu_flag=True, output_size=10):
        super(GClassifier, self).__init__()

        tmp_module_list = []

        for c_index in range(1, len(layer_list)):
            tmp_module_list.append(nn.Linear(layer_list[c_index-1], layer_list[c_index]))
            if relu_flag:
                tmp_module_list.append(nn.ReLU())
            else:
                tmp_module_list.append(nn.LeakyReLU())
        tmp_module_list.append(nn.Linear(layer_list[-1], output_size))

        self.model = nn.Sequential(*tmp_module_list)

        # Initialize the weight via Xavier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

    def forward(self, img):
        validity = self.model(img)
        return validity


# General Vector Space Discriminator, the input would be N-D vector
class GDiscriminator(nn.Module):
    def __init__(self, layer_list, relu=True, norm=None, out_dim=1):
        super(GDiscriminator, self).__init__()

        tmp_module_list = []

        for c_index in range(1, len(layer_list)):
            # Add standard linear layer
            tmp_module_list.append(nn.Linear(layer_list[c_index-1], layer_list[c_index]))

            # Add Normalization layer if specified
            if norm is not None:
                if norm == 'batch':
                    tmp_module_list.append(nn.BatchNorm1d(layer_list[c_index]))
                elif norm == 'instance':
                    tmp_module_list.append(nn.InstanceNorm1d(layer_list[c_index]))
                else:
                    raise Exception("Unsupported Normalization Method Detected")

            # Add activation layer (ReLU or LeakyReLU)
            if relu:
                tmp_module_list.append(nn.ReLU())
            else:
                tmp_module_list.append(nn.LeakyReLU())

        # Add last layer, which outputs the logit of final decision (Input of a sigmoid func or something like that)
        tmp_module_list.append(nn.Linear(layer_list[-1], out_dim))

        self.model = nn.Sequential(*tmp_module_list)

        # Initialize the weight via Xavier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

    def forward(self, img):
        validity = self.model(img)
        return validity

# General Image Space Discriminator,
