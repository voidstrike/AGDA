import numpy as np
import os
import params
import sys
import torch
import datetime
import PTransformer as ptf

from torchvision.datasets import MNIST, SVHN
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets
from torch import nn
from copy import deepcopy

from LinearAE import LinearAE
from ConvAE import LeNetAE28, LeNetAE32, ExLeNetAE28, ExAlexNet
from FCNN import GDiscriminator, GClassifier
from usps import USPS
from torchvision.models import alexnet


# Aux_func s.t. passing input img to target model via its type
def forwardByModelType(in_model, in_vec, p_size=params.input_img_size, p_channel=params.input_img_channel):
    if not isinstance(in_model, LinearAE):
        code, rec = in_model(in_vec.view(-1, p_channel, p_size, p_size))
    else:
        code, rec = in_model(in_vec.view(in_vec.shape[0], -1))
    return code, rec


# Aux_func s.t. returns specific model via the value of hidden dimension
def getModelByDimension(t_dim, office_flag=False):
    ae, clf, dis = None, None, None

    if not office_flag:
        if t_dim == 100:
            clf = GClassifier(params.DEFAULT_CLF_100)
            dis = GDiscriminator(params.DEFAULT_DIS_100)
            ae = LeNetAE28()
        elif t_dim == 400:
            clf = GClassifier(params.DEFAULT_CLF_400)
            dis = GDiscriminator(params.DEFAULT_DIS_400)
            ae = LeNetAE28()
        elif t_dim == 800:
            clf = GClassifier(params.DEFAULT_CLF_800)
            dis = GDiscriminator(params.DEFAULT_DIS_800)
            ae = ExLeNetAE28()
        elif t_dim == 500:
            clf = GClassifier(params.DEFAULT_CLF_500)
            dis = GDiscriminator(params.DEFAULT_DIS_500)
            ae = ExLeNetAE28(True)
        elif t_dim == -1:
            clf = GClassifier(params.DEFAULT_CLF_400)
            dis = GDiscriminator(params.DEFAULT_DIS_400)
            ae = LeNetAE32()
    else:
        # Deal with OFFICE-31 data set, using AlexNet, VGG-16, ResNet?
        # TODO
        tmp_model = alexnet(pretrained=True)

        if t_dim == -2:
            clf = GClassifier(params.DEFAULT_IMG_DIS, output_size=31)
            dis = GDiscriminator(params.DEFAULT_IMG_CLF)
            ae = ExAlexNet(tmp_model.features.float())
        pass

    return ae, clf, dis


# Auxiliary function that return the number of correct prediction
def getHitCount(t_label, p_label):
        _, p_label = p_label.max(1)
        num_correct = (t_label == p_label).sum().item()
        return num_correct


# Auxiliary function that returns the AE Loss and Classifier Loss
# in_dl -- input dataset / ae_criterion -- BCELoss or BCEWithLogitsLoss
def getModelPerformance(in_dl, in_ae, in_clf, ae_criterion):
    ae_loss, clf_acc, instance_count = .0, .0, in_dl.dataset.__len__()

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
        if params.input_img_size == 28:
            data_set = MNIST(root_path + '/../data/mnist',  train=train, transform=ptf.directly_tfs, download=True)
        else:
            data_set = MNIST(root_path + '/../data/mnist',  train=train, transform=ptf.tfs_32, download=True)
    elif ds_name == "usps":
        data_set = USPS(root_path + '/../data', train=train, transform=ptf.directly_tfs, download=True)
    elif ds_name == "svhn":
        if params.input_img_size == 28:
            data_set = SVHN(root_path + '/../data/svhn', split='train' if train else 'test',
                             transform=ptf.tfs_28_gray, download=True)
        else:
            data_set = SVHN(root_path + '/../data/svhn', split='train' if train else 'test',
                             transform=ptf.tfs_32_gray, download=True)

    elif ds_name == "amazon" or ds_name == "webcam" or ds_name == "dslr":
        tmp_path = root_path + '/../data/office/'
        tmp_path += 'train/' if train else 'test/'
        data_set = datasets.ImageFolder(tmp_path + ds_name + "/", transform=ptf.tfs_224)
    else:
        raise Exception("Unsupported Dataset")

    r_sampler = RandomSampler(data_set, replacement=False)
    target_dl = DataLoader(data_set, batch_size=params.batch_size, shuffle=True)
    if train:
        target_dl_fusion = DataLoader(data_set, sampler=r_sampler, batch_size=params.fusion_size)

    return target_dl, target_dl_fusion


def getMMD(sdl, sm, tdl, tm, hd):
    s_mean = getDisMean(sdl, sm, hd)
    t_mean = getDisMean(tdl, tm, hd)
    res = (s_mean - t_mean) ** 2
    return res.sum().item()


def getDisMean(data_loader, tfs_model, h_dim):
    if h_dim == -1:
        h_dim = 400
    res = torch.zeros(1, h_dim)
    res = res.cuda() if torch.cuda.is_available() else res
    instance_count = 0.
    for feature, _ in data_loader:
        feature = feature.cuda() if torch.cuda.is_available() else feature
        instance_count += feature.shape[0]
        ex_feature, _ = tfs_model(feature)
        res += ex_feature.sum(dim=-2)
    return res / instance_count


def main(load_model=False, hidden_dim=100, cuda_flag=False):
    if hidden_dim not in params.dim_set:
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
    source_ae, source_clf, target_dis = getModelByDimension(hidden_dim)

    if load_model:
        source_ae.load_state_dict(torch.load(root_path + '/../modeinfo/source_ae_' + params.source_data_set + '.pt'))
        source_ae.eval()
        source_clf.load_state_dict(torch.load(root_path + '/../modeinfo/source_clf_' + params.source_data_set + '.pt'))
        source_clf.eval()

    criterion_ae = nn.MSELoss(reduction='sum')      # General Loss of AE -- sum MSE
    criterion_clf = nn.CrossEntropyLoss()           # General Loss of classifier -- CEL
    criterion_gan = nn.BCEWithLogitsLoss()          # Auxiliary loss for GAN (Discriminator)

    src_optimizer = torch.optim.Adam(list(source_ae.parameters()) + list(source_clf.parameters()),
                                     lr=params.clf_learning_rate,
                                     weight_decay=2.5e-5,
                                     betas=(.5, .999))

    if cuda_flag:
        source_ae = source_ae.cuda()
        source_clf = source_clf.cuda()

    if not load_model:
        # Train AE & CLF from scratch
        instance_count = source_train_data.dataset.__len__()
        for step in range(params.clf_train_iter):
            ae_loss_iter, clf_loss_iter, train_acc_iter = .0, .0, .0

            for features, label in source_train_data:
                if cuda_flag:
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

    if cuda_flag:
        target_ae = target_ae.cuda()
        target_dis = target_dis.cuda()

    # Train target AE and Discriminator
    currentDT = datetime.datetime.now()
    currentDT = str(currentDT.strftime("%m-%d-%H-%M"))
    tmp_log = open(root_path + "/../log/experiment_log_" + currentDT + ".txt", 'w')
    for step in range(params.tag_train_iter):
        data_zip = enumerate(zip(source_train_data_fusion, target_train_data_fusion))
        for _, ((src_f, _), (tgt_f, _)) in data_zip:

            src_f = Variable(src_f.view(src_f.shape[0], -1))
            src_valid = Variable(torch.ones(src_f.size(0), 1, dtype=torch.float32))
            tgt_f = Variable(tgt_f.view(tgt_f.shape[0], -1))
            tgt_valid = Variable(torch.ones(tgt_f.size(0), 1, dtype=torch.float32))
            tgt_fake = Variable(torch.zeros(tgt_f.size(0), 1, dtype=torch.float32))

            if cuda_flag:
                src_f = src_f.cuda()
                src_valid = src_valid.cuda()
                tgt_f = tgt_f.cuda()
                tgt_valid = tgt_valid.cuda()
                tgt_fake = tgt_fake.cuda()

            src_code, src_rec = forwardByModelType(source_ae, src_f)
            tgt_code, tgt_rec = forwardByModelType(target_ae, tgt_f)

            for d_step in range(params.d_steps):
                optimizer_D.zero_grad()

                src_domain_label = target_dis(src_code)
                tgt_domain_label = target_dis(tgt_code.detach())

                over_all_domain_label = torch.cat((src_domain_label, tgt_domain_label), 0)
                over_all_domain_valid = torch.cat((src_valid, tgt_fake), 0)

                d_loss = criterion_gan(over_all_domain_label, over_all_domain_valid)

                if d_step != params.d_steps - 1:
                    d_loss.backward(retain_graph=True)
                else:
                    d_loss.backward()

                optimizer_D.step()

            for g_step in range(params.g_steps):
                optimizer_G.zero_grad()
                tgt_code, tgt_rec = forwardByModelType(target_ae, tgt_f)
                tgt_domain_label = target_dis(tgt_code)

                loss_tgt_rec = criterion_ae(tgt_f, tgt_rec)
                loss_tgt_gen = criterion_gan(tgt_domain_label, tgt_valid)

                loss_total = params.target_ae_weight * loss_tgt_rec + params.target_fusion_weight * loss_tgt_gen

                loss_total.backward()
                optimizer_G.step()

        # Test the accuracy after this iteration
        ae_loss_train, train_acc = getModelPerformance(target_train_data, target_ae, source_clf, criterion_ae)
        ae_loss_test, test_acc = getModelPerformance(target_test_data, target_ae, source_clf, criterion_ae)

        print('Epoch: {}, AE Loss train: {:.6f}, Clf Acc Train: {:.6f}, AE Loss Target: {:.6f}, Clf Acc Target: {:.6f}'
              .format(step, ae_loss_train, train_acc, ae_loss_test, test_acc))
        tmp_log.write(
            '{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}\n'.format(step, ae_loss_train, train_acc, ae_loss_test, test_acc))

        # Auxiliary metric -- MMD
        c_mmd = getMMD(source_test_data, source_ae, target_test_data, target_ae, hidden_dim)
        print('Current MMD is: {:.6f}'.format(c_mmd))

    tmp_log.close()


if __name__ == '__main__':
    load_flag = False
    hidden_dim = 100
    if len(sys.argv) > 2:
        load_flag = bool(sys.argv[1] == 'True')
        hidden_dim = int(sys.argv[2])
    cuda_flag = torch.cuda.is_available()

    main(load_flag, hidden_dim, cuda_flag)
