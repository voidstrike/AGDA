from torch import nn
from torch.nn import init


# General version of classifier -- Configure the network structure in params.py
class GClassifier(nn.Module):
    def __init__(self, layer_list, relu_flag=True):
        super(GClassifier, self).__init__()

        tmp_module_list = []

        for c_index in range(1, len(layer_list)):
            tmp_module_list.append(nn.Linear(layer_list[c_index-1], layer_list[c_index]))
            if relu_flag:
                tmp_module_list.append(nn.ReLU())
            else:
                tmp_module_list.append(nn.LeakyReLU())
        tmp_module_list.append(nn.Linear(layer_list[-1], 10))

        self.model = nn.Sequential(*tmp_module_list)

        # Initialize the weight via Xavier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

    def forward(self, img):
        validity = self.model(img)
        return validity


# General version of discriminator -- Configure the network structure in params.py
class GDiscriminator(nn.Module):
    def __init__(self, layer_list, relu_flag=True):
        super(GDiscriminator, self).__init__()

        tmp_module_list = []

        for c_index in range(1, len(layer_list)):
            tmp_module_list.append(nn.Linear(layer_list[c_index-1], layer_list[c_index]))
            if relu_flag:
                tmp_module_list.append(nn.ReLU())
            else:
                tmp_module_list.append(nn.LeakyReLU())
        tmp_module_list.append(nn.Linear(layer_list[-1], 1))

        self.model = nn.Sequential(*tmp_module_list)

        # Initialize the weight via Xavier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

    def forward(self, img):
        validity = self.model(img)
        return validity
