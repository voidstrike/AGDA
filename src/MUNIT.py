# Assumption: Sharing latent content space, No-sharing latent style space
# from model.ResNet_UNIT import *
from model.Net_MUNIT import *

import os
import itertools
import torchvision.transforms as transforms
import datetime
import time
import random
import numpy as np

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

from torch.optim import lr_scheduler


def get_all_data_loader(root_path, transformer, batch_size=1):
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


def sample_cycle_image(current_epoch, dl1, dl2, E1, E2, G1, G2):
    # Randomly select one img each domain then perform the transfer process
    f1 = dl1.dataset[random.randint(0, dl1.dataset.__len__() - 1)][0].unsqueeze(0)
    f2 = dl2.dataset[random.randint(0, dl2.dataset.__len__() - 1)][0].unsqueeze(0)

    if torch.cuda.is_available():
        X1 = Variable(f1.type(torch.Tensor).cuda())
        X2 = Variable(f2.type(torch.Tensor).cuda())
    else:
        X1 = Variable(f1.type(torch.Tensor))
        X2 = Variable(f2.type(torch.Tensor))

    C1, S1 = E1(X1)
    C2, S2 = E2(X2)

    fake_X1 = G1(C2, X1)
    fake_X2 = G2(C1, X2)

    img_sample = torch.cat((X1.data, fake_X2.data, X2.data, fake_X1.data), 0)
    save_image(img_sample, "images/%s/%s.png" % ("test2", current_epoch), nrow=5, normalize=True)


def main():
    # Basic program setting
    input_shape = (3, 256, 256)
    learning_rate = 1e-4
    dim = 64  # Number of filters of the first Conv layer
    style_dim = 8
    n_downsample = 2
    shared_dim = dim * 2 ** n_downsample
    n_epochs = 2000
    scheduler_flag = False

    global_batch_size = 1

    # Initialize generator and discriminator
    E1 = Encoder(dim=dim, n_downsample=n_downsample, n_residual=4, style_dim=style_dim)
    E2 = Encoder(dim=dim, n_downsample=n_downsample, n_residual=4, style_dim=style_dim)

    G1 = Decoder(dim=dim, n_upsample=n_downsample, n_residual=4, style_dim=style_dim)
    G2 = Decoder(dim=dim, n_upsample=n_downsample, n_residual=4, style_dim=style_dim)

    D1 = MultiDiscriminator(input_shape)
    D2 = MultiDiscriminator(input_shape)

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
    lambda_gan = 1  # GAN
    lambda_rec_img = 10  # Image Reconstruction Loss
    lambda_rec_sty = 1  # Style Reconstruction Loss
    lambda_rec_con = 1  # Content Reconstruction Loss
    lambda_cyc = 100  # Style augmented cycle consistency loss

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(E1.parameters(), E2.parameters(), G1.parameters(), G2.parameters()),
        lr=learning_rate,
        betas=(.5, .999),
    )
    optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=learning_rate, betas=(.5, .999), weight_decay=1e-4)
    optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=learning_rate, betas=(.5, .999), weight_decay=1e-4)

    # Learning rate update schedulers -- Currently set to constant
    lr_scheduler_G = None
    lr_scheduler_D1 = None
    lr_scheduler_D2 = None

    if scheduler_flag:
        lr_scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=1000, gamma=.5, last_epoch=-1)
        lr_scheduler_D1 = lr_scheduler.StepLR(optimizer_D1, step_size=1000, gamma=.5, last_epoch=-1)
        lr_scheduler_D2 = lr_scheduler.StepLR(optimizer_D2, step_size=1000, gamma=.5, last_epoch=-1)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # Temporary Image Transformation
    t_trans_ = transforms.Compose([
        transforms.Resize(int(input_shape[1] * 1.12)),
        transforms.RandomCrop((input_shape[1], input_shape[2])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    traX, traY, tesX, tesY = get_all_data_loader(os.getcwd(), t_trans_, 1)

    prev_time = time.time()

    # -------------------------------
    #  Train Encoders and Generators (Decoders)
    # -------------------------------
    valid, fake = 1, 0

    for current_epoch in range(n_epochs):
        for pivot, ((train_img_X, _), (train_img_Y, _)) in enumerate(zip(traX, traY)):
            if torch.cuda.is_available():
                X1 = Variable(train_img_X.type(Tensor).cuda())
                X2 = Variable(train_img_Y.type(Tensor).cuda())
                S1 = Variable(torch.randn(X1.size(0), style_dim, 1, 1).type(Tensor))
                S2 = Variable(torch.randn(X1.size(0), style_dim, 1, 1).type(Tensor))
            else:
                X1 = Variable(train_img_X.type(Tensor))
                X2 = Variable(train_img_Y.type(Tensor))
                S1 = Variable(torch.randn(X1.size(0), style_dim, 1, 1).type(Tensor))
                S2 = Variable(torch.randn(X1.size(0), style_dim, 1, 1).type(Tensor))


            optimizer_G.zero_grad()

            # Get shared latent representation
            CON1, STY1 = E1(X1)
            CON2, STY2 = E2(X2)

            # Reconstruct images
            rec_X1 = G1(CON1, STY1)
            rec_X2 = G2(CON2, STY2)

            # Translate images
            fake_X1 = G1(CON2, STY1)
            fake_X2 = G2(CON1, STY1)

            # Perform cycle translation
            C_CON2, C_STY_1 = E1(fake_X1)
            C_CON1, C_STY_2 = E2(fake_X2)

            X121 = G1(C_CON1, STY1)
            X212 = G2(C_CON2, STY2)

            # Losses
            # Losses that discriminator can distinguish fake & real images
            loss_GAN_1 = lambda_gan * D1.compute_loss(fake_X1, valid)
            loss_GAN_2 = lambda_gan * D1.compute_loss(fake_X2, valid)

            # Image Reconstruction Loss
            loss_rec_img_1 = lambda_rec_img * criterion_pixel(rec_X1, X1)
            loss_rec_img_2 = lambda_rec_img * criterion_pixel(rec_X2, X2)

            # Style Reconstruction Loss
            loss_rec_sty_1 = lambda_rec_sty * criterion_pixel(C_STY_1, S1)
            loss_rec_sty_2 = lambda_rec_sty * criterion_pixel(C_STY_2, S2)

            # Content Reconstruction Loss
            loss_rec_con_1 = lambda_rec_con * criterion_pixel(C_CON1, CON1.detach())
            loss_rec_con_2 = lambda_rec_con * criterion_pixel(C_CON2, CON2.detach())

            # Circle reconstruction loss
            loss_cyc_1 =lambda_cyc * criterion_pixel(X121, X1)
            loss_cyc_2 = lambda_cyc* criterion_pixel(X212, X2)

            # Total loss
            loss_G = (
                    loss_GAN_1
                    + loss_GAN_2
                    + loss_rec_img_1
                    + loss_rec_img_2
                    + loss_rec_sty_1
                    + loss_rec_sty_2
                    + loss_rec_con_1
                    + loss_rec_con_2
                    + loss_cyc_1
                    + loss_cyc_2
            )

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator 1
            # -----------------------

            optimizer_D1.zero_grad()

            loss_D1 = D1.compute_loss(X1, valid) + D1.compute_loss(fake_X1.detach(), fake)
            loss_D1.backward()

            optimizer_D1.step()

            # -----------------------
            #  Train Discriminator 2
            # -----------------------

            optimizer_D2.zero_grad()

            loss_D2 = D2.compute_loss(X2, valid) + D2.compute_loss(fake_X2.detach(), fake)
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
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
                % (current_epoch, n_epochs, pivot, len(traX), (loss_D1 + loss_D2).item(), loss_G.item(), time_left)
            )

            # Update learning rates
        if lr_scheduler_G is not None:
            lr_scheduler_G.step()
        if lr_scheduler_D1 is not None:
            lr_scheduler_D1.step()
        if lr_scheduler_D2 is not None:
            lr_scheduler_D2.step()

        if current_epoch % 100 == 0:
            sample_cycle_image(current_epoch, traX, traY, E1, E2, G1, G2)


if __name__ == "__main__":
    main()