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
from ConvAE import  LeNetAE28, LeNetAE32
from FCNN import LinearClf100, Discriminator100, LinearClf400, Discriminator400
from usps import USPS
from GSVHN import GSVHN


def setPartialTrainable(target_model, num_layer):
    # Train only part of the model
    if num_layer != 0:
        ct = 0
        if isinstance(target_model, LinearAE):
            for eachLayer in target_model.encoder:
                if isinstance(eachLayer, torch.nn.Linear) and ct < num_layer:
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
    instance_count = in_dl.dataset.__len__()
    for features, label in in_dl:
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

    # Initialize models for source domain
    source_ae = LeNetAE28()
    if hidden_dim == 100:
        source_clf = LinearClf100()
    elif hidden_dim == 400:
        source_clf = LinearClf400()

    # Initialize models for target domain
    # target_ae = LeNetAE()
    target_dis = Discriminator100() if hidden_dim == 100 else Discriminator400()

    if load_model:
        source_ae.load_state_dict(torch.load(root_path + '/../modeinfo/source_ae_' + params.source_data_set + '.pt'))
        source_ae.eval()
        source_clf.load_state_dict(torch.load(root_path + '/../modeinfo/source_clf_' + params.source_data_set + '.pt'))
        source_clf.eval()

    criterion_ae = nn.MSELoss(reduction='sum')      # General Loss of AE -- sum MSE
    criterion_clf = nn.CrossEntropyLoss()           # General Loss of classifier -- CEL
    criterion_gan = nn.BCELoss()                    # Auxiliary loss for GAN (Discriminator)

    src_optimizer = torch.optim.Adam(list(source_ae.parameters()) + list(source_clf.parameters()), lr=1e-3)

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
                floss = loss_ae + loss_clf
                # floss = loss_clf

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

        torch.save(source_ae.state_dict(), root_path + '/../modeinfo/source_ae_' + params.source_data_set + '.pt')
        torch.save(source_clf.state_dict(), root_path + '/../modeinfo/source_clf_' + params.source_data_set + '.pt')

    # Show the performance of trained model in source domain, train & test set
    ae_loss_train, train_acc = getModelPerformance(source_train_data, source_ae, source_clf, criterion_ae)
    ae_loss_test, test_acc = getModelPerformance(source_test_data, source_ae, source_clf, criterion_ae)

    print('Trained Model AE Loss train: {:.6f}, Clf Acc Train: {:.6f}, AE Loss Target: {:.6f}, Clf Acc Target: {:.6f}'
          .format(ae_loss_train, train_acc, ae_loss_test, test_acc))

    # Models for target domain
    target_ae = deepcopy(source_ae)  # Copy from Source AE
    # target_ae = LeNetAE28()
    setPartialTrainable(target_ae, params.num_disable_layer)
    # target_ae = LinearAE((28*28, 256, 64, 16, 8), None)

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
                
                # print("R : {:.6f}, F : {:.6f}".format(real_loss.item(), fake_loss.item()))

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
                # floss = g_loss
                # print(floss.item())
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
