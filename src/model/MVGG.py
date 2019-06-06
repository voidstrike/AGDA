from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19


# Modified VGG-19 that can output activations of selected layers
class MultiVGG19(nn.Module):
    def __init__(self, pool='max'):
        super(MultiVGG19, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = dict()
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

    # Copy the weight & bias from other trained VGG-19 model (Same structure in torch model zoo)
    def weight_from_model_zoo(self, tgt):
        fe = tgt.features
        self.conv1_1.weight, self.conv1_1.bias = fe[0].weight, fe[0].bias
        self.conv1_2.weight, self.conv1_2.bias = fe[2].weight, fe[2].bias
        self.conv2_1.weight, self.conv2_1.bias = fe[5].weight, fe[5].bias
        self.conv2_2.weight, self.conv2_2.bias = fe[7].weight, fe[7].bias
        self.conv3_1.weight, self.conv3_1.bias = fe[10].weight, fe[10].bias
        self.conv3_2.weight, self.conv3_2.bias = fe[12].weight, fe[12].bias
        self.conv3_3.weight, self.conv3_3.bias = fe[14].weight, fe[14].bias
        self.conv3_4.weight, self.conv3_4.bias = fe[16].weight, fe[16].bias
        self.conv4_1.weight, self.conv4_1.bias = fe[19].weight, fe[19].bias
        self.conv4_2.weight, self.conv4_2.bias = fe[21].weight, fe[21].bias
        self.conv4_3.weight, self.conv4_3.bias = fe[23].weight, fe[23].bias
        self.conv4_4.weight, self.conv4_4.bias = fe[25].weight, fe[25].bias
        self.conv5_1.weight, self.conv5_1.bias = fe[28].weight, fe[28].bias
        self.conv5_2.weight, self.conv5_2.bias = fe[30].weight, fe[30].bias
        self.conv5_3.weight, self.conv5_3.bias = fe[32].weight, fe[32].bias
        self.conv5_4.weight, self.conv5_4.bias = fe[34].weight, fe[34].bias
        pass


def main():
    # Test code to copy weight from exist model
    pivot_vgg = vgg19(pretrained=True)
    tgt_model = MultiVGG19(pivot_vgg)
    print(tgt_model.state_dict())


if __name__ == "__main__":
    main()
