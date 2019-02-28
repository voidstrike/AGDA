import numpy as np
import torch
import sys
from torchvision.datasets import MNIST, CIFAR10
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms as tfs
from torchvision.utils import save_image
from torch import nn
from usps import USPS, get_usps
#from usps16 import USPS16, get_usps
from torch.nn import init
from copy import deepcopy

import matplotlib.pyplot as plt


# MNIST -- toy net
class BasicAE(nn.Module):
    def __init__(self):
        super(BasicAE, self).__init__()

        # Encoder Structure
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 64),
            nn.LeakyReLU(True),
            nn.Linear(64, 16),
            nn.LeakyReLU(True),
            nn.Linear(16, 8)
        )

        # Decoder Structure
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(True),
            nn.Linear(16, 64),
            nn.LeakyReLU(True),
            nn.Linear(64, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

    def forward(self, x):
        code = self.encoder(x)
        rec = self.decoder(code)

        return code, rec


class BasicAE2(nn.Module):

    def __init__(self):
        super(BasicAE2, self).__init__()

        # Encoder Structure
        self.encoder = nn.Sequential(
            nn.Linear(16 * 16, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 64),
            nn.LeakyReLU(True),
            nn.Linear(64, 16),
            nn.LeakyReLU(True),
            nn.Linear(16, 8)
        )

        # Decoder Structure
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(True),
            nn.Linear(16, 64),
            nn.LeakyReLU(True),
            nn.Linear(64, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 16 * 16),
            nn.Sigmoid()
        )

    def forward(self, x):
        code = self.encoder(x)
        rec = self.decoder(code)

        return code, rec


# CIFAR10
class BasicAE3(nn.Module):
    def __init__(self):
        super(BasicAE3, self).__init__()

        # Encoder Structure
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 64)
        )

        # Decoder Structure
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, 32 * 32 * 3),
            nn.Tanh()
        )

    def forward(self, x):
        code = self.encoder(x)
        rec = self.decoder(code)

        return code, rec


class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()

        # Encoder Network
        # Output Shape = (Original_Shape - Kernel + 2 * Padding) / Stride
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        code = self.encoder(x)
        rec = self.decoder(code)
        return code, rec


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # Create Sequential Model
        self.model = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
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

#  GAN - Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)

        return validity


def to_img(x):
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x

im_tfs = tfs.Compose([
    tfs.ToTensor()
])



def main(load_model=False):
    # Get source domain data
    train_set = MNIST('./data/mnist', transform=im_tfs, download=True)
    train_data = DataLoader(train_set, batch_size=128, shuffle=True)

    # Get target domain data
    target_train_data = get_usps(True)

    # Models for source domain
    source_ae = BasicAE()
    source_clf = Classifier()
    if load_model:
        source_ae.load_state_dict(torch.load('./modeinfo/source_ae.pt'))
        source_ae.eval()
        source_clf.load_state_dict(torch.load('./modeinfo/source_clf.pt'))
        source_clf.eval()



    criterion_ae = nn.MSELoss(reduction='sum')  # General loss for AE -- sum MSE
    criterion_clf = nn.CrossEntropyLoss()  # General Loss for classifier -- CEL
    criterion_gan = nn.BCELoss()           # Auxiliary loss for GAN

    optimizer = torch.optim.Adam(list(source_ae.parameters()) + list(source_clf.parameters()), lr=1e-3)

    # optimizer = torch.optim.Adam(source_ae.parameters(), lr=1e-3)

    if torch.cuda.is_available():
        source_ae = source_ae.cuda()
        source_clf = source_clf.cuda()

    if not load_model:
        # Train the AutoEncoder and Classifier for source domain if not loaded
        for step in range(200):
            ae_loss = 0.0
            clf_loss = 0.0
            train_acc = 0.0

            for features, label in train_data:
                if torch.cuda.is_available():
                    features = Variable(features.view(features.shape[0], -1).cuda())
                    # Used USPS
                    # label = label.squeeze(1)
                    label = Variable(label.cuda())

                else:
                    features = Variable(features.view(features.shape[0], -1))
                    # Used USPS
                    # label = label.squeeze(1)
                    label = Variable(label)

                source_code, source_rec = source_ae(features)
                label_pred = source_clf(source_code)

                loss_ae = criterion_ae(features, source_rec)
                loss_clf = criterion_clf(label_pred, label)
                floss = loss_ae + loss_clf

                optimizer.zero_grad()
                floss.backward()
                optimizer.step()

                ae_loss += loss_ae.item()
                clf_loss += loss_clf.item()

                _, pred = label_pred.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / features.shape[0]
                train_acc += acc

            print('epoch: {}, AutoEncoder Loss: {:.6f}, Classifier Loss: {:.6f}, Train Acc: {:.6f}'
                  .format(step, ae_loss / len(train_data), clf_loss / len(train_data),
                          train_acc / len(train_data)))

        torch.save(source_ae.state_dict(), './modeinfo/source_ae.pt')
        torch.save(source_clf.state_dict(), './modeinfo/source_clf.pt')

    # Models for target domain
    target_ae = deepcopy(source_ae)  # Copy from Source AE
    target_dis = Discriminator()

    #  Disable part of the AE
    i = 0
    for eachLayer in target_ae.encoder:
        if isinstance(eachLayer, torch.nn.Linear) and i < 2:
            eachLayer.requires_grad = False
            i += 1

    optimizer_G = torch.optim.Adam(target_ae.parameters(), lr=1e-4)
    optimizer_D = torch.optim.Adam(target_dis.parameters(), lr=1e-3)

    if torch.cuda.is_available():
        target_ae = target_ae.cuda()
        target_dis = target_dis.cuda()

    valid_placeholder = Variable(torch.from_numpy(np.ones((128, 1), dtype='float32')), requires_grad=False)
    fake_placeholder = Variable(torch.from_numpy(np.zeros((128, 1), dtype='float32')), requires_grad=False)
    if torch.cuda.is_available():
        valid_placeholder = valid_placeholder.cuda()
        fake_placeholder = fake_placeholder.cuda()

    # Train target AE and Discriminator
    for step in range(1000):
        #  Train discriminator
        for features, label in target_train_data:
            if features.shape[0] != 128:
                continue
            if torch.cuda.is_available():
                features = Variable(features.view(features.shape[0], -1).cuda())
                label = label.squeeze(1)
                label = Variable(label.cuda())
            else:
                features = Variable(features.view(features.shape[0], -1))
                label = label.squeeze(1)
                label = Variable(label)

            # Sample for source domain -- Real Image
            t_key = np.random.randint(train_data.__len__() - 1)
            sampler = SubsetRandomSampler(list(range(t_key * 128, (t_key + 1) * 128)))
            real_loader = DataLoader(train_set, sampler=sampler, shuffle=False, batch_size=128)

            target_code, target_rec = target_ae(features)

            real_code = None
            for s_feature, _ in real_loader:
                s_feature = torch.reshape(s_feature, (-1, 28 * 28))
                if torch.cuda.is_available():
                    s_feature = s_feature.cuda()
                real_code, _ = source_ae(s_feature)

            optimizer_D.zero_grad()

            dis_res_real_code = target_dis(real_code)
            tmp, _ = target_ae(features)
            dis_res_fake_code = target_dis(tmp)
            real_loss = criterion_gan(dis_res_real_code, valid_placeholder)
            fake_loss = criterion_gan(dis_res_fake_code, fake_placeholder)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train AE (Generator + decoder)
            optimizer_G.zero_grad()

            gen_res_fake_code = target_dis(target_code)
            g_loss = criterion_gan(gen_res_fake_code, valid_placeholder)
            ae_loss = criterion_ae(features, target_rec)
            floss = g_loss + ae_loss
            floss.backward()
            optimizer_G.step()

        # Test the accuracy
        ae_loss = 0.0
        train_acc = 0.0
        for features, label in target_train_data:
            if torch.cuda.is_available():
                features = Variable(features.view(features.shape[0], -1).cuda())
                label = label.squeeze(1)
                label = Variable(label.cuda())
                valid_placeholder = valid_placeholder.cuda()
                fake_placeholder = fake_placeholder.cuda()
            else:
                features = Variable(features.view(features.shape[0], -1))
                label = label.squeeze(1)
                label = Variable(label)

            target_code, target_rec = target_ae(features)
            loss_ae = criterion_ae(features, target_rec)

            ae_loss += loss_ae.item()

            label_pred = source_clf(target_code)

            _, pred = label_pred.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / features.shape[0]
            train_acc += acc

        print('epoch: {}, AutoEncoder Loss: {:.6f}, Train Acc: {:.6f}'
              .format(step, ae_loss / len(target_train_data), train_acc / len(target_train_data)))


if __name__ == '__main__':
    Load_flag = False
    if len(sys.argv) > 1:
        Load_flag = bool(sys.argv[1])
    main(Load_flag)



