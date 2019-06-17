# Latent Code Sharing Assumption

from model.ResNet_UNIT import *

import math
import os
import itertools
import torchvision.transforms as transforms
import datetime
import time
import sys
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torch.autograd import Variable

from torch.optim import lr_scheduler


def get_all_data_loader(root_path, transformer, batch_size=32):
    root_path = os.path.join(root_path, '../data/unit')

    train_x = get_data_loader_folder(os.path.join(root_path, 'trainX/'), transformer, batch_size)
    test_x = get_data_loader_folder(os.path.join(root_path, 'testX/'), transformer, batch_size, shuffle=False)

    train_y = get_data_loader_folder(os.path.join(root_path, 'trainY/'), transformer, batch_size)
    test_y = get_data_loader_folder(os.path.join(root_path, 'testY/'), transformer, batch_size, shuffle=False)

    return train_x, train_y, test_x, test_y


def get_data_loader_folder(input_folder, transformer, batch_size, shuffle=True):
    dataset = ImageFolder(input_folder, transform=transformer)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader


def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss


def main():
    # Specify input shape
    input_shape = (3, 256, 256)
    dim = 64  # Number of filters in the first Conv layer
    learning_rate = 1e-4
    n_downsample = 2
    shared_dim = dim * 2 ** n_downsample
    n_epochs = 200
    epoch = 0
    decay_epoch = 100
    global_batch_size = 1
    # Initialize generator and discriminator

    shared_E = ResidualBlock(features=shared_dim)
    E1 = Encoder(dim=dim, n_downsample=n_downsample, shared_block=shared_E)
    E2 = Encoder(dim=dim, n_downsample=n_downsample, shared_block=shared_E)
    shared_G = ResidualBlock(features=shared_dim)

    G1 = Decoder(dim=dim, n_upsample=n_downsample, shared_block=shared_G)
    G2 = Decoder(dim=dim, n_upsample=n_downsample, shared_block=shared_G)

    D1 = Discriminator(input_shape)
    D2 = Discriminator(input_shape)

    criterion_GAN = torch.nn.MSELoss()
    criterion_pixel = torch.nn.L1Loss()

    if torch.cuda.is_available():
        E1 = E1.cuda()
        E2 = E2.cuda()
        G1 = G1.cuda()
        G2 = G2.cuda()
        D1 = D1.cuda()
        D2 = D2.cuda()
        criterion_GAN.cuda()
        criterion_pixel.cuda()

    # Loss weights
    lambda_0 = 10  # GAN
    lambda_1 = 0.1  # KL (encoded images)
    lambda_2 = 100  # ID pixel-wise
    lambda_3 = 0.1  # KL (encoded translated images)
    lambda_4 = 100  # Cycle pixel-wise

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(E1.parameters(), E2.parameters(), G1.parameters(), G2.parameters()),
        lr=learning_rate,
        betas=(.5, .999),
    )
    optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=learning_rate, betas=(.5, .999))
    optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=learning_rate, betas=(.5, .999))

    # Learning rate update schedulers -- Currently set to constant
    lr_scheduler_G = None
    lr_scheduler_D1 = None
    lr_scheduler_D2 = None

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # Temporary Image Transformation
    transforms_ = [
        transforms.Resize(int(input_shape[1] * 1.12)),
        transforms.RandomCrop((input_shape[1], input_shape[2])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    traX, traY, tesX, tesY = get_all_data_loader(os.getcwd(), transforms, 32)

    prev_time = time.time()

    # -------------------------------
    #  Train Encoders and Generators (Decoders)
    # -------------------------------

    for current_epoch in range(n_epochs):
        for pivot, (train_img_X, train_img_Y) in enumerate(zip(traX, traY)):
            if torch.cuda.is_available():
                X1 = Variable(train_img_X.type(Tensor).cuda())
                X2 = Variable(train_img_Y.type(Tensor).cuda())
                valid = Variable(Tensor(np.ones((X1.size(0), *D1.output_shape))).cuda(), requires_grad=False)
                fake = Variable(Tensor(np.zeros((X1.size(0), *D1.output_shape))).cuda(), requires_grad=False)
            else:
                X1 = Variable(train_img_X.type(Tensor))
                X2 = Variable(train_img_Y.type(Tensor))
                valid = Variable(Tensor(np.ones((X1.size(0), *D1.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((X1.size(0), *D1.output_shape))), requires_grad=False)


            optimizer_G.zero_grad()

            # Get shared latent representation
            mu1, Z1 = E1(X1)
            mu2, Z2 = E2(X2)

            # Reconstruct images
            rec_X1 = G1(Z1)
            rec_X2 = G2(Z2)

            # Translate images
            fake_X1 = G1(Z2)
            fake_X2 = G2(Z1)

            # Cycle translation
            mu1_, Z1_ = E1(fake_X1)
            mu2_, Z2_ = E2(fake_X2)
            cyc_X1 = G1(Z2_)
            cyc_X2 = G2(Z1_)

            # Losses
            loss_GAN_1 = lambda_0 * criterion_GAN(D1(fake_X1), valid)
            loss_GAN_2 = lambda_0 * criterion_GAN(D2(fake_X2), valid)
            loss_KL_1 = lambda_1 * compute_kl(mu1)
            loss_KL_2 = lambda_1 * compute_kl(mu2)
            loss_ID_1 = lambda_2 * criterion_pixel(rec_X1, X1)
            loss_ID_2 = lambda_2 * criterion_pixel(rec_X2, X2)
            loss_KL_1_ = lambda_3 * compute_kl(mu1_)
            loss_KL_2_ = lambda_3 * compute_kl(mu2_)
            loss_cyc_1 = lambda_4 * criterion_pixel(cyc_X1, X1)
            loss_cyc_2 = lambda_4 * criterion_pixel(cyc_X2, X2)

            # Total loss
            loss_G = (
                    loss_KL_1
                    + loss_KL_2
                    + loss_ID_1
                    + loss_ID_2
                    + loss_GAN_1
                    + loss_GAN_2
                    + loss_KL_1_
                    + loss_KL_2_
                    + loss_cyc_1
                    + loss_cyc_2
            )

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator 1
            # -----------------------

            optimizer_D1.zero_grad()

            loss_D1 = criterion_GAN(D1(X1), valid) + criterion_GAN(D1(fake_X1.detach()), fake)

            loss_D1.backward()
            optimizer_D1.step()

            # -----------------------
            #  Train Discriminator 2
            # -----------------------

            optimizer_D2.zero_grad()

            loss_D2 = criterion_GAN(D2(X2), valid) + criterion_GAN(D2(fake_X2.detach()), fake)

            loss_D2.backward()
            optimizer_D2.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = current_epoch * len(traX) + pivot * global_batch_size
            batches_left = n_epochs * len(traX) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
                % (epoch, n_epochs, pivot, len(traX), (loss_D1 + loss_D2).item(), loss_G.item(), time_left)
            )

            # Update learning rates
        if lr_scheduler_G is not None:
            lr_scheduler_G.step()
        if lr_scheduler_D1 is not None:
            lr_scheduler_D1.step()
        if lr_scheduler_D2 is not None:
            lr_scheduler_D2.step()


if __name__ == "__main__":
    main()
