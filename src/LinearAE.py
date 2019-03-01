from torch import nn
from torch.nn import init


# Linear AE class
class LinearAE(nn.Module):

    def __init__(self, encoder_set, decoder_set, activation='LeakyReLU', rescale='Sigmoid', initizer='xavier'):
        super(LinearAE, self).__init__()
        self.encoder = nn.Sequential(*self.__generateEncoder(encoder_set, activation))
        if decoder_set is None:
            self.decoder = nn.Sequential(*self.__generateDecoder(encoder_set, activation, rescale, True))
        else:
            self.decoder = nn.Sequential(*self.__generateDecoder(decoder_set, activation, rescale, False))

        if initizer is not None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    if initizer == 'xavier':
                        init.xavier_uniform_(module.weight)
                    elif initizer == 'normal':
                        init.normal_(module.weight, mean=0, std=1)
                    elif initizer == 'uniform':
                        init.uniform_(module.weight, a=0, b=1)
                    else:
                        raise Exception("Unknown Initialization Function Presented")

    def __generateEncoder(self, encoder_set, activation):
        layer_set = []
        for i in range(len(encoder_set) - 1):
            layer_set.append(nn.Linear(encoder_set[i], encoder_set[i+1]))
            if activation == 'LeakyReLU':
                layer_set.append(nn.LeakyReLU(True))
            else:
                raise Exception("Unknown Activation Function Presented")
        layer_set.append(nn.Linear(encoder_set[-2], encoder_set[-1]))
        return layer_set

    def __generateDecoder(self, decoder_set, activation, rescale, flag=False):
        layer_set = []
        if flag:
            decoder_set.reverse()

        for i in range(len(decoder_set) - 1):
            layer_set.append(nn.Linear(decoder_set[i], decoder_set[i + 1]))
            if activation == 'LeakyReLU':
                layer_set.append(nn.LeakyReLU(True))
            else:
                raise Exception("Unknown Activation Function Presented")
        layer_set.append(nn.Linear(decoder_set[-2], decoder_set[-1]))

        if rescale is None:
            pass
        elif rescale == 'Sigmoid':
            layer_set.append(nn.Sigmoid())
        elif rescale == 'Tanh':
            layer_set.append(nn.Tanh())
        else:
            raise Exception("Unknown Rescale Function Presented")

        return layer_set

    def forward(self, x):
        code = self.encoder(x)
        rec = self.decoder(code)
        return code, rec
