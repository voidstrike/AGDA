from torch import nn
from torch.nn import init


# Linear AE class
class ConvAE(nn.Module):

    def __init__(self, encoder_set, decoder_set, initizer='xavier'):
        super(ConvAE, self).__init__()
        self.encoder = nn.Sequential(*self.__generateEncoder(encoder_set))
        self.decoder = nn.Sequential(*self.__generateDecoder(decoder_set))

        if initizer is not None:
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                    if initizer == 'xavier':
                        init.xavier_uniform_(module.weight)
                    elif initizer == 'normal':
                        init.normal_(module.weight, mean=0, std=1)
                    elif initizer == 'uniform':
                        init.uniform_(module.weight, a=0, b=1)
                    else:
                        raise Exception("Unknown Initialization Function Presented")

    #  encoder_set should be a list of 9-tuple -- (input_channels, output_channels, kernel_size, stride, padding,
    #  activation, pool, p_block, p_stride)
    #  default stride = 1 and default padding = 0
    def __generateEncoder(self, encoder_set):
        layer_set = []
        for i in range(len(encoder_set)):
            layer_set.append(nn.Conv2d(encoder_set[0], encoder_set[1], encoder_set[2],
                                       stride=encoder_set[3], padding=encoder_set[4]))
            if encoder_set[5] == 'LeakyReLU':
                layer_set.append(nn.LeakyReLU(True))
            elif encoder_set[5] == 'ReLU':
                layer_set.append(nn.ReLU(True))
            else:
                raise Exception("Unknown Activation Function Presented")

            if encoder_set[6] == 'Max':
                layer_set.append(nn.MaxPool2d(encoder_set[7], stride=encoder_set[8]))

        return layer_set

    # decoder_set should be a list of 6-tuple -- (input_channels, output_channels, kernel_size, stride, padding,
    # follow functions)
    def __generateDecoder(self, decoder_set):
        layer_set = []
        for i in range(len(decoder_set)):
            layer_set.append(nn.ConvTranspose2d(decoder_set[0], decoder_set[1], decoder_set[2],
                                       stride=decoder_set[3], padding=decoder_set[4]))
            if decoder_set[5] == 'LeakyReLU':
                layer_set.append(nn.LeakyReLU(True))
            elif decoder_set[5] == 'ReLU':
                layer_set.append(nn.ReLU(True))
            elif decoder_set[5] == 'Tanh':
                layer_set.append(nn.Tanh())
            else:
                raise Exception("Unknown Activation Function Presented")

        return layer_set

    def forward(self, x):
        code = self.encoder(x)
        rec = self.decoder(code)
        return code, rec

class TupleGenerator(object):
    def __init__(self):
        pass

    def getEncoderTuple(self, ic, oc, ks, stride=1, padding=0, act='LeakyReLU', pool='Max', pkernel=2, pstride=1):
        res = (ic, oc, ks, stride, padding, act, pool, pkernel, pstride)
        return res

    def generateDecoderTuple(self, input_channales, output_channels, kernel_size, stride=1, padding=0, func='ReLU'):
        res = (input_channales, output_channels, kernel_size, stride, padding, func)
        return res
