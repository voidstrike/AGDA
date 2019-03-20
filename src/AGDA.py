import numpy as np
import os
import params
import sys
import torch
import datetime

from torchvision.datasets import MNIST
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms as tfs
from torch import nn
from copy import deepcopy

from LinearAE import LinearAE
from ConvAE import LeNetAE28, ExLeNetAE28
from FCNN import LinearClf100, Discriminator100, LinearClf400, Discriminator400
from FCNN import LinearClf800, Discriminator800
from usps import USPS
from GSVHN import GSVHN


# Auxiliary function that forward input through in_model once
def forwardByModelType(in_model, in_vec, psize=28, pchannel=1):
    if not isinstance(in_model, LinearAE):
        code, rec = in_model(in_vec.view(-1, pchannel, psize, psize))
    else:
        code, rec = in_model(in_vec)
    return code, rec


# Auxiliary function that return the number of correct prediction
def getHitCount(tlabel, plabel):
    _, plabel = plabel.max(1)
    num_correct = (tlabel == plabel).sum().item()
    return num_correct


# Auxiliary function that returns the AE Loss and Classifier Loss
# in_dl -- input dataset / ae_criterion -- BCELoss or BCEWithLogitsLoss
def getModelPerformance(in_dl, in_ae, in_clf, ae_criterion):
    ae_loss = 0.0
    clf_acc = 0.0
    instance_count = in_dl.dataset.__len__()
    for features, label in in_dl:

        features = Variable(features.view(features.shape[0], -1))
        label = Variable(label)

        if torch.cuda.is_available():
            features = features.cuda()
            label = label.cuda()

        target_code, target_rec = forwardByModelType(in_ae, features)

        ae_loss_batch = ae_criterion(features, target_rec)
        ae_loss += ae_loss_batch.item()

        label_pred = in_clf(target_code)

        clf_acc += getHitCount(label, label_pred)

    return ae_loss / instance_count, clf_acc / instance_count


# Auxiliary fuction that returns the dataLoader via name
# It will return an extra fusion dataloader iff the train flag is True
def getDataLoader(ds_name, root_path, train=True):
    target_dl, target_dl_fusion = None, None

    # Get data set by their name
    if ds_name == "mnist":
        data_set = MNIST(root_path + '/../data/mnist',  train=train, transform=im_tfs, download=True)
    elif ds_name == "usps":
        data_set = USPS(root_path + '/../data', train=train, transform=im_tfs, download=True)
    elif ds_name == "svhn":
        data_set = GSVHN(root_path + '/../data/svhn', split='train' if train else 'test', transform=o_tfs, download=True)
    else:
        raise Exception("Unsupported Dataset")

    r_sampler = RandomSampler(data_set, replacement=True, num_samples=params.fusion_size)
    target_dl = DataLoader(data_set, batch_size=params.batch_size, shuffle=True)
    if train:
        target_dl_fusion = DataLoader(data_set, sampler=r_sampler, batch_size=params.fusion_size)

    return target_dl, target_dl_fusion


# Auxiliary function that convert tensor back to img
def to_img(x):
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x


# Auxiliary torch.transforms object that convert img to tensor
im_tfs = tfs.Compose([
    tfs.ToTensor()
])

o_tfs = tfs.Compose([
    tfs.CenterCrop(28),
    tfs.ToTensor()
])


def main(load_model=False, hidden_dim=100):
    if hidden_dim != 100 and hidden_dim != 400:
        raise Exception("Unsupported Hidden Dimension")
    root_path = os.getcwd()

    # Get source domain data
    source_train_data, source_train_data_fusion = getDataLoader(params.source_data_set, root_path, True)
    source_test_data, _ = getDataLoader(params.source_data_set, root_path, False)

    # Get target domain data
    target_train_data, target_train_data_fusion = getDataLoader(params.target_data_set, root_path, True)
    target_test_data, _ = getDataLoader(params.target_data_set, root_path, False)

    # Initialize source classifier and target discriminator
    # source_ae = LeNetAE28()
    source_ae = ExLeNetAE28()
    if hidden_dim == 100:
        source_clf = LinearClf100()
        target_dis = Discriminator100()
    elif hidden_dim == 400:
        source_clf = LinearClf400()
        target_dis = Discriminator400()
    elif hidden_dim == 800:
        source_clf = LinearClf800()
        target_dis = Discriminator800()

    if load_model:
        source_ae.load_state_dict(torch.load(root_path + '/../modeinfo/source_ae_' + params.source_data_set + '.pt'))
        source_ae.eval()
        source_clf.load_state_dict(torch.load(root_path + '/../modeinfo/source_clf_' + params.source_data_set + '.pt'))
        source_clf.eval()

    criterion_ae = nn.MSELoss(reduction='sum')      # General Loss of AE -- sum MSE
    criterion_clf = nn.CrossEntropyLoss()           # General Loss of classifier -- CEL
    criterion_gan = nn.BCEWithLogitsLoss()          # Auxiliary loss for GAN (Discriminator)

    src_optimizer = torch.optim.Adam(list(source_ae.parameters()) + list(source_clf.parameters()), lr=params.clf_learning_rate)

    if torch.cuda.is_available():
        source_ae = source_ae.cuda()
        source_clf = source_clf.cuda()

    if not load_model:
        # Train AE & CLF from scratch
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
                floss = params.source_ae_weight * loss_ae + params.source_clf_weight * loss_clf

                src_optimizer.zero_grad()
                floss.backward()
                src_optimizer.step()

                ae_loss_iter += loss_ae.item()
                clf_loss_iter += loss_clf.item()

                train_acc_iter += getHitCount(label, label_predict)

            print('Epoch: {}, AutoEncoder Loss: {:.6f}, Classifier Loss: {:.6f}, Train Acc: {:.6f}'
                  .format(step, ae_loss_iter / instance_count, clf_loss_iter / instance_count,
                          train_acc_iter / instance_count))

        torch.save(source_ae.state_dict(), root_path + '/../modeinfo/source_ae_' + params.source_data_set + '.pt')
        torch.save(source_clf.state_dict(), root_path + '/../modeinfo/source_clf_' + params.source_data_set + '.pt')

    # Show the performance of trained model in source domain, train & test set
    ae_loss_train, train_acc = getModelPerformance(source_train_data, source_ae, source_clf, criterion_ae)
    ae_loss_test, test_acc = getModelPerformance(source_test_data, source_ae, source_clf, criterion_ae)

    print('Trained Model AE Loss train: {:.6f}, Clf Acc Train: {:.6f}, AE Loss Target: {:.6f}, Clf Acc Target: {:.6f}'
          .format(ae_loss_train, train_acc, ae_loss_test, test_acc))

    # Models for target domain
    target_ae = deepcopy(source_ae)  # Copy from Source AE -- Fine tuning method
    target_ae.setPartialTrainable(params.num_disable_layer)

    optimizer_G = torch.optim.Adam(target_ae.parameters(), lr=params.g_learning_rate)
    optimizer_D = torch.optim.Adam(target_dis.parameters(), lr=params.d_learning_rate)

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
    currentDT = datetime.datetime.now()
    currentDT = str(currentDT.strftime("%m-%d-%H-%M"))
    tmp_log = open(root_path + "/../log/experiment_log_" + currentDT + ".txt", 'w')
    for step in range(params.tag_train_iter):
        for features, _ in target_train_data_fusion:
            if torch.cuda.is_available():
                features = Variable(features.view(features.shape[0], -1).cuda())
            else:
                features = Variable(features.view(features.shape[0], -1))

            real_code = None
            for s_feature, _ in source_train_data_fusion:
                if torch.cuda.is_available():
                    s_feature = s_feature.cuda()
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

                floss = params.target_fusion_weight * g_loss + params.target_ae_weight * ae_loss
                floss.backward()

                optimizer_G.step()

        # Test the accuracy after this iteration
        if step % 200 == 0:
            ae_loss_train, train_acc = getModelPerformance(target_train_data, target_ae, source_clf, criterion_ae)
            ae_loss_test, test_acc = getModelPerformance(target_test_data, target_ae, source_clf, criterion_ae)

            print('Epoch: {}, AE Loss train: {:.6f}, Clf Acc Train: {:.6f}, AE Loss Target: {:.6f}, Clf Acc Target: {:.6f}'
                .format(step, ae_loss_train, train_acc, ae_loss_test, test_acc))
            tmp_log.write('{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}\n'.format(step, ae_loss_train, train_acc, ae_loss_test, test_acc))

    tmp_log.close()


if __name__ == '__main__':
    load_flag = False
    hidden_dim = 100
    if len(sys.argv) > 2:
        load_flag = bool(sys.argv[1] == 'True')
        hidden_dim = int(sys.argv[2])

    main(load_flag, hidden_dim)
