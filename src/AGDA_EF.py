import numpy as np
import os
import params
import sys
import torch

from torchvision.datasets import MNIST, SVHN
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms as tfs
from torch import nn

from LinearAE import LinearAE
from ConvAE import ConvAE, LeNetAE28, LeNetAE32
from FCNN import LinearClf, Discriminator
from usps import USPS


def setPartialTrainable(target_model, num_layer):
    # Train only part of the model
    if num_layer != 0:
        ct = 0
        if isinstance(target_model, LinearAE):
            for eachLayer in target_model.encoder:
                if isinstance(eachLayer, torch.nn.Linear) and ct < num_layer:
                    eachLayer.requires_grad = False
                    ct += 1
        elif isinstance(target_model, ConvAE):
            for eachLayer in target_model.encoder:
                if isinstance(eachLayer, torch.nn.Conv2d) and ct < num_layer:
                    eachLayer.requires_grad = False
                    ct += 1
        elif isinstance(target_model, LeNetAE28) or isinstance(target_model, LeNetAE32):
            for eachLayer in target_model.encoder_cnn:
                if isinstance(eachLayer, torch.nn.Conv2d) and ct < num_layer:
                    eachLayer.requires_grad = False
                    ct += 1
    return target_model


def getModelPerformance(in_dl, in_ae, in_clf, ae_criterion):
    ae_loss = 0.0
    clf_acc = 0.0
    instance_count = 0.0
    for features, label in in_dl:
        instance_count += features.shape[0]
        if torch.cuda.is_available():
            features = Variable(features.view(features.shape[0], -1).cuda())
            label = Variable(label.cuda())
        else:
            features = Variable(features.view(features.shape[0], -1))
            label = Variable(label)

        target_code, target_rec = forwardByModelType(in_ae, features)

        ae_loss_batch = ae_criterion(features, target_rec)
        ae_loss += ae_loss_batch.item()

        label_pred = in_clf(target_code)

        _, pred = label_pred.max(1)
        num_correct = (pred == label).sum().item()
        clf_acc += num_correct

    return ae_loss / instance_count, clf_acc / instance_count


def forwardByModelType(in_model, in_vec):
    if not isinstance(in_model, LinearAE):
        code, rec = in_model(in_vec.view(-1, 1, 28, 28))
    else:
        code, rec = in_model(in_vec)
    return code, rec


def getDataLoader(ds_name, root_path, train=True):
    # Initially, training data loader & fusion data loader are None
    target_dl, target_dl_fusion = None, None

    # Get data set by their name
    if ds_name == "mnist":
        data_set = MNIST(root_path + '/data/mnist', train=train, transform=im_tfs, download=True)
    elif ds_name == "usps":
        data_set = USPS(root_path + '/data', train=train, transform=im_tfs, download=True)
    elif ds_name == "svhn":
        data_set = SVHN(root_path + '/data/svhn', split='train' if train else 'test', transform=im_tfs, download=True)
    else:
        raise Exception("Unsupported Dataset")

    r_sampler = RandomSampler(data_set, replacement=True, num_samples=params.fusion_size)
    target_dl = DataLoader(data_set, batch_size=params.batch_size, shuffle=True)
    if train:
        target_dl_fusion = DataLoader(data_set, sampler=r_sampler, batch_size=params.fusion_size)

    return target_dl, target_dl_fusion


# Auxiliary func that convert tensor back to img
def to_img(x):
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x


# Auxiliary torch.transforms object that convert img to tensor
im_tfs = tfs.Compose([
    tfs.ToTensor()
])


# Early Fusion version
def main(load_model=False):
    root_path = os.getcwd()

    # Get source domain data
    source_train_data, source_train_data_fusion = getDataLoader(params.source_data_set, root_path, True)

    # Get target domain data
    target_train_data, target_train_data_fusion = getDataLoader(params.target_data_set, root_path, True)
    target_test_data, _ = getDataLoader('usps', root_path, False)

    # Initialize models for source domain
    source_ae = LeNetAE28()
    source_clf = LinearClf()

    # Initialize models for target domain
    target_ae = LeNetAE28()
    target_dis = Discriminator()

    if load_model:
        source_ae.load_state_dict(torch.load(root_path + '/../modeinfo/source_ae.pt'))
        source_ae.eval()
        source_clf.load_state_dict(torch.load(root_path + '/../modeinfo/source_clf.pt'))
        source_clf.eval()

    criterion_ae = nn.MSELoss(reduction='sum')  # General Loss of AE -- sum MSE
    criterion_clf = nn.CrossEntropyLoss()  # General Loss of classifier -- CEL
    criterion_gan = nn.BCELoss()  # Auxiliary loss for GAN (Discriminator)

    src_optimizer = torch.optim.Adam(list(source_ae.parameters()) + list(source_clf.parameters()), lr=1e-3)
    optimizer_G = torch.optim.Adam(target_ae.parameters(), lr=1e-3)
    optimizer_D = torch.optim.Adam(target_dis.parameters(), lr=1e-3)

    valid_placeholder_fusion = Variable(torch.from_numpy(np.ones((params.fusion_size, 1), dtype='float32')),
                                        requires_grad=False)
    fake_placeholder_fusion = Variable(torch.from_numpy(np.zeros((params.fusion_size, 1), dtype='float32')),
                                       requires_grad=False)

    if torch.cuda.is_available():
        source_ae = source_ae.cuda()
        source_clf = source_clf.cuda()
        target_ae = target_ae.cuda()
        target_dis = target_dis.cuda()
        valid_placeholder_fusion = valid_placeholder_fusion.cuda()
        fake_placeholder_fusion = fake_placeholder_fusion.cuda()

    if not load_model:
        # Train AE & CLF from scratch
        # Fuse the domain in hidden space each iteration
        instance_count = source_train_data.dataset.__len__()
        for step in range(params.clf_train_iter):
            ae_loss_iter, clf_loss_iter, train_acc_iter = .0, .0, .0

            for features, label in source_train_data:
                if torch.cuda.is_available():
                    features = Variable(features.view(features.shape[0], -1).cuda())
                    label = Variable(label.cuda())

                else:
                    features = Variable(features.view(features.shape[0], -1))
                    label = Variable(label)

                source_code, source_rec = forwardByModelType(source_ae, features)
                label_predict = source_clf(source_code)

                loss_ae = criterion_ae(features, source_rec)
                loss_clf = criterion_clf(label_predict, label)
                floss = loss_ae + loss_clf

                src_optimizer.zero_grad()
                floss.backward()
                src_optimizer.step()

                ae_loss_iter += loss_ae.item()
                clf_loss_iter += loss_clf.item()

                _, pred = label_predict.max(1)
                num_correct = (pred == label).sum().item()
                train_acc_iter += num_correct

            print('Epoch: {}, AutoEncoder Loss: {:.6f}, Classifier Loss: {:.6f}, Train Acc: {:.6f}'
                  .format(step, ae_loss_iter / instance_count, clf_loss_iter / instance_count,
                          train_acc_iter / instance_count))

            # Fusion fusion_steps times
            for f_step in range(params.fusion_steps):
                for features, _ in target_train_data_fusion:
                    if torch.cuda.is_available():
                        features = Variable(features.view(features.shape[0], -1).cuda())
                    else:
                        features = Variable(features.view(features.shape[0], -1))

                    real_code = None
                    for s_feature, _ in source_train_data_fusion:
                        s_feature = s_feature.cuda() if torch.cuda.is_available() else s_feature
                        real_code, _ = forwardByModelType(source_ae, s_feature)
                    assert real_code is not None

                    # Train Discriminator k times
                    fake_code, _ = forwardByModelType(target_ae, features)
                    for d_step in range(params.d_steps):
                        optimizer_D.zero_grad()

                        dis_res_real_code = target_dis(real_code)
                        dis_res_fake_code = target_dis(fake_code)

                        real_loss = criterion_gan(dis_res_real_code, valid_placeholder_fusion)
                        fake_loss = criterion_gan(dis_res_fake_code, fake_placeholder_fusion)
                        d_loss = (real_loss + fake_loss) / 2
                        if d_step != params.d_steps - 1:
                            d_loss.backward(retain_graph=True)
                        else:
                            d_loss.backward()

                        optimizer_D.step()

                    # Train Generator & Decoder k' times
                    for g_step in range(params.g_steps):
                        target_code, target_rec = forwardByModelType(target_ae, features)

                        optimizer_G.zero_grad()
                        gen_res_fake_code = target_dis(target_code)

                        g_loss = criterion_gan(gen_res_fake_code, valid_placeholder_fusion)
                        ae_loss = criterion_ae(features, target_rec)
                        floss = g_loss + ae_loss
                        floss.backward()

                        optimizer_G.step()


            # Test the result after fusion
            ae_loss_train, train_acc = getModelPerformance(target_train_data, target_ae, source_clf, criterion_ae)
            ae_loss_target, test_acc = getModelPerformance(target_test_data, target_ae, source_clf, criterion_ae)

            print('Epoch: {}, AELoss_tra: {:.6f}, CAcc_tra: {:.6f}, AELoss_tar: {:.6f}, CAcc_tar: {:.6f}'
                  .format(step, ae_loss_train, train_acc, ae_loss_target, test_acc))

        torch.save(source_ae.state_dict(), root_path + '/../modeinfo/source_ae.pt')
        torch.save(source_clf.state_dict(), root_path + '/../modeinfo/source_clf.pt')


if __name__ == '__main__':
    Load_flag = False
    if len(sys.argv) > 1:
        Load_flag = bool(sys.argv[1])
    main(Load_flag)


