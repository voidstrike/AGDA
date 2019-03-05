import numpy as np
import torch
import sys, os.path
import os
from LinearAE import LinearAE
from ConvAE import ConvAE
from FCNN import LinearClf, Discriminator
from torchvision.datasets import MNIST
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms as tfs
from torch import nn
from usps import get_usps
import params
from copy import deepcopy


def setPartialTrainable(target_model, num_layer):
    # Train only part of the model
    if num_layer != 0:
        ct = 0
        for eachLayer in target_model.encoder:
            if isinstance(eachLayer, torch.nn.Linear) and ct < num_layer:
                eachLayer.requires_grad = False
                ct += 1
    return target_model


def to_img(x):
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x

im_tfs = tfs.Compose([
    tfs.ToTensor()
])


def main(load_model=False):

    root_path = os.getcwd()

    # Get source domain data
    train_set = MNIST(root_path + '/data/mnist', transform=im_tfs, download=True)
    train_data = DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
    train_data_fusion = DataLoader(train_set, batch_size=params.fusion_size, shuffle=True)

    # Get target domain data
    target_train_data = get_usps(root_path + '/data', True)

    # Models for source domain
    # source_ae = BasicAE()
    # source_ae = LinearAE((28*28, 256, 64, 16, 8), None)  # Generate source AE
    source_ae = ConvAE()
    source_clf = LinearClf()

    if load_model:
        source_ae.load_state_dict(torch.load(root_path + '/modeinfo/source_ae.pt'))
        source_ae.eval()
        source_clf.load_state_dict(torch.load(root_path + '/modeinfo/source_clf.pt'))
        source_clf.eval()

    criterion_ae = nn.MSELoss(reduction='sum')      # General Loss of AE -- sum MSE
    criterion_clf = nn.CrossEntropyLoss()           # General Loss of classifier -- CEL
    criterion_gan = nn.BCELoss()                    # Auxiliary loss for GAN (Discriminator)

    sae_opt = torch.optim.Adam(list(source_ae.parameters()) + list(source_clf.parameters()), lr=1e-3)

    if torch.cuda.is_available():
        source_ae = source_ae.cuda()
        source_clf = source_clf.cuda()

    if not load_model:
        # Train AE & CLF from scratch
        for step in range(params.clf_train_iter):
            ae_loss_iter = 0.0
            clf_loss_iter = 0.0
            train_acc_iter = 0.0

            for features, label in train_data:
                if torch.cuda.is_available():
                    features = Variable(features.view(features.shape[0], -1).cuda())
                    label = Variable(label.cuda())

                else:
                    features = Variable(features.view(features.shape[0], -1))
                    label = Variable(label)

                if isinstance(source_ae, ConvAE):
                    source_code, source_rec = source_ae(features.view(-1, 1, 28, 28))
                    source_rec = source_rec.view(-1, 28 * 28)
                    source_code = source_code.flatten(start_dim=1)
                else:
                    source_code, source_rec = source_ae(features)

                label_pred = source_clf(source_code)

                loss_ae = criterion_ae(features, source_rec)
                loss_clf = criterion_clf(label_pred, label)
                floss = loss_ae + loss_clf

                sae_opt.zero_grad()
                floss.backward()
                sae_opt.step()

                ae_loss_iter += loss_ae.item()
                clf_loss_iter += loss_clf.item()

                _, pred = label_pred.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / features.shape[0]
                train_acc_iter += acc

            print('epoch: {}, AutoEncoder Loss: {:.6f}, Classifier Loss: {:.6f}, Train Acc: {:.6f}'
                  .format(step, ae_loss_iter / len(train_data), clf_loss_iter / len(train_data),
                          train_acc_iter / len(train_data)))

        torch.save(source_ae.state_dict(), root_path + '/../modeinfo/source_ae.pt')
        torch.save(source_clf.state_dict(), root_path + '/../modeinfo/source_clf.pt')

    # Models for target domain
    target_ae = deepcopy(source_ae)  # Copy from Source AE
    setPartialTrainable(target_ae, 2)
    # target_ae = LinearAE((28*28, 256, 64, 16, 8), None)
    target_dis = Discriminator()

    optimizer_G = torch.optim.Adam(target_ae.parameters(), lr=1e-4)
    optimizer_D = torch.optim.Adam(target_dis.parameters(), lr=1e-3)

    valid_placeholder_fusion = Variable(torch.from_numpy(np.ones((params.fusion_size, 1), dtype='float32')),
                                 requires_grad=False)
    fake_placeholder_fusion = Variable(torch.from_numpy(np.zeros((params.fusion_size, 1), dtype='float32')),
                                requires_grad=False)

    if torch.cuda.is_available():
        target_ae = target_ae.cuda()
        target_dis = target_dis.cuda()
        valid_placeholder_fusion = valid_placeholder_fusion.cuda()
        fake_placeholder_fusion = fake_placeholder_fusion.cuda()

    # Train target AE and Discriminator
    for step in range(params.tag_train_iter):
        #  Train discriminator
        for features, _ in target_train_data:
            if features.shape[0] != params.fusion_size:
                continue
            if torch.cuda.is_available():
                features = Variable(features.view(features.shape[0], -1).cuda())
            else:
                features = Variable(features.view(features.shape[0], -1))

            # Sample for source domain -- Real Image
            t_key = np.random.randint(train_data_fusion.__len__() - 1)
            sampler = SubsetRandomSampler(list(range(t_key * params.fusion_size, (t_key + 1) * params.fusion_size)))
            real_loader = DataLoader(train_set, sampler=sampler, shuffle=False, batch_size=params.fusion_size)

            if isinstance(target_ae, ConvAE):
                target_code, target_rec = target_ae(features.view(-1, 1, 28, 28))
                target_code = target_code.flatten(start_dim=1)
                target_rec = target_rec.view(-1, 28 * 28)
            else:
                target_code, target_rec = target_ae(features)

            real_code = None
            for s_feature, _ in real_loader:
                # s_feature = torch.reshape(s_feature, (-1, 28 * 28))
                if torch.cuda.is_available():
                    s_feature = s_feature.cuda()

                if isinstance(source_ae, ConvAE):
                    real_code, _ = source_ae(s_feature.view(-1, 1, 28, 28))
                    real_code = real_code.flatten(start_dim=1)
                else:
                    real_code, _ = source_ae(s_feature)

            # print(real_code.shape)
            optimizer_D.zero_grad()
            dis_res_real_code = target_dis(real_code)
            if isinstance(target_ae, ConvAE):
                tmp, _ = target_ae(features.view(-1, 1, 28, 28))
                tmp = tmp.flatten(start_dim=1)
            else:
                tmp, _ = target_ae(features)

            dis_res_fake_code = target_dis(tmp)
            real_loss = criterion_gan(dis_res_real_code, valid_placeholder_fusion)
            fake_loss = criterion_gan(dis_res_fake_code, fake_placeholder_fusion)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train AE (Generator + decoder)
            optimizer_G.zero_grad()

            gen_res_fake_code = target_dis(target_code)
            g_loss = criterion_gan(gen_res_fake_code, valid_placeholder_fusion)
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
                label = Variable(label.cuda())
            else:
                features = Variable(features.view(features.shape[0], -1))
                label = Variable(label)

            if isinstance(target_ae, ConvAE):
                target_code, target_rec = target_ae(features.view(-1, 1, 28, 28))
                target_code = target_code.flatten(start_dim=1)
                target_rec = target_rec.view(-1, 28 * 28)
            else:
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



