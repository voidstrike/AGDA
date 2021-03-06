# Latent Code Sharing Assumption

from model.ResNet_UNIT import *

import os
import itertools
import torchvision.transforms as transforms
import datetime
import time
import random
import argparse

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

from torch.optim import lr_scheduler


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        tensor[:, 0, :, :] = tensor[:, 0, :, :].mul(self.std[0]).add_(self.mean[0])
        tensor[:, 1, :, :] = tensor[:, 1, :, :].mul(self.std[1]).add_(self.mean[1])
        tensor[:, 2, :, :] = tensor[:, 2, :, :].mul(self.std[2]).add_(self.mean[2])

        return tensor


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_all_data_loader(root_path, transformer_tr, transformer_te, batch_size=1):
    root_path = os.path.join(root_path, '../data/unit')

    train_x = get_data_loader_folder(os.path.join(root_path, 'trainX/'), transformer_tr, batch_size)
    test_x = get_data_loader_folder(os.path.join(root_path, 'testX/'), transformer_te, batch_size, shuffle=False)

    train_y = get_data_loader_folder(os.path.join(root_path, 'trainY/'), transformer_tr, batch_size)
    test_y = get_data_loader_folder(os.path.join(root_path, 'testY/'), transformer_te, batch_size, shuffle=False)

    return train_x, train_y, test_x, test_y


def get_data_loader_folder(input_folder, transformer, batch_size, shuffle=True):
    dataset = ImageFolderWithPaths(input_folder, transform=transformer)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader


def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss


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

    _, Z1 = E1(X1)
    _, Z2 = E2(X2)

    fake_X1 = G1(Z2)
    fake_X2 = G2(Z1)

    img_sample = torch.cat((X1.data, fake_X2.data, X2.data, fake_X1.data), 0)
    save_image(img_sample, "images/%s/%s.png" % ("test", current_epoch), nrow=5, normalize=True)


def main(conf):
    # Basic program setting
    input_shape = (3, 256, 256)
    learning_rate = 1e-4
    dim = 64  # Number of filters of the first Conv layer
    n_downsample = 2
    shared_dim = dim * 2 ** n_downsample
    n_epochs = 100
    scheduler_flag = False

    global_batch_size = 1

    # Initialize generator and discriminator
    shared_E = ResidualBlock(features=shared_dim, drop_out=.5)
    E1 = Encoder(dim=dim, n_downsample=n_downsample, shared_block=shared_E)
    E2 = Encoder(dim=dim, n_downsample=n_downsample, shared_block=shared_E)

    shared_G = ResidualBlock(features=shared_dim, drop_out=.5)
    G1 = Decoder(dim=dim, n_upsample=n_downsample, shared_block=shared_G)
    G2 = Decoder(dim=dim, n_upsample=n_downsample, shared_block=shared_G)

    D1 = Discriminator(input_shape)
    D2 = Discriminator(input_shape)

    criterion_GAN = torch.nn.MSELoss()
    criterion_pixel = torch.nn.L1Loss()

    if opt.test:
        n_epochs = 0
        E1.load_state_dict(torch.load(conf.model_dir + 'E1.pt'))
        E2.load_state_dict(torch.load(conf.model_dir + 'E2.pt'))
        G1.load_state_dict(torch.load(conf.model_dir + 'G1.pt'))
        G2.load_state_dict(torch.load(conf.model_dir + 'G2.pt'))
        D1.load_state_dict(torch.load(conf.model_dir + 'D1.pt'))
        D2.load_state_dict(torch.load(conf.model_dir + 'D2.pt'))

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
        transforms.Resize(286),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_trans_ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    traX, traY, tesX, tesY = get_all_data_loader(os.getcwd(), t_trans_, test_trans_, 1)

    prev_time = time.time()

    # -------------------------------
    #  Train Encoders and Generators (Decoders)
    # -------------------------------

    for current_epoch in range(n_epochs):
        for pivot, ((train_img_X, _, _), (train_img_Y, _, _)) in enumerate(zip(traX, traY)):
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

            # Perform cycle translation
            mu1_, Z1_ = E1(fake_X1)
            mu2_, Z2_ = E2(fake_X2)

            cyc_X1 = G1(Z2_)
            cyc_X2 = G2(Z1_)

            # Losses
            # Losses that discriminator can distinguish fake & real images
            loss_GAN_1 = lambda_0 * criterion_GAN(D1(fake_X1), valid)
            loss_GAN_2 = lambda_0 * criterion_GAN(D2(fake_X2), valid)

            # KL loss for reconstruction
            loss_KL_1 = lambda_1 * compute_kl(mu1)
            loss_KL_2 = lambda_1 * compute_kl(mu2)

            # Reconstruction Loss
            loss_ID_1 = lambda_2 * criterion_pixel(rec_X1, X1)
            loss_ID_2 = lambda_2 * criterion_pixel(rec_X2, X2)

            # KL loss for circle consistency
            loss_KL_1_ = lambda_3 * compute_kl(mu1_)
            loss_KL_2_ = lambda_3 * compute_kl(mu2_)

            # Circle reconstruction loss
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
            pass
            # sample_cycle_image(current_epoch, traX, traY, E1, E2, G1, G2)
    if not opt.test:
        torch.save(E1.state_dict(), conf.model_dir + 'E1.pt')
        torch.save(E2.state_dict(), conf.model_dir + 'E2.pt')
        torch.save(G1.state_dict(), conf.model_dir + 'G1.pt')
        torch.save(G2.state_dict(), conf.model_dir + 'G2.pt')
        torch.save(D1.state_dict(), conf.model_dir + 'D1.pt')
        torch.save(D2.state_dict(), conf.model_dir + 'D2.pt')

#   ------------------------------------TEST
    denorm = UnNormalize((.5, .5, .5), (.5, .5, .5))
    for pivot, ((test_img_X, _, x_path), (test_img_Y, _, y_path)) in enumerate(zip(tesX, tesY)):
        if torch.cuda.is_available():
            X1 = Variable(test_img_X.type(Tensor).cuda())
            X2 = Variable(test_img_Y.type(Tensor).cuda())
        else:
            X1 = Variable(test_img_X.type(Tensor))
            X2 = Variable(test_img_Y.type(Tensor))

        # Get shared latent representation
        _, Z1 = E1(X1)
        _, Z2 = E2(X2)

        # Translate images
        fake_X1 = G1(Z2)
        fake_X2 = G2(Z1)

        fake_A = denorm(fake_X1)[:1, :, :, :]
        fake_A_name = x_path[0].split('/')[-1]
        fake_B = denorm(fake_X2)[:1, :, :, :]
        fake_B_name = y_path[0].split('/')[-1]
        save_image(fake_A, conf.output_dir + 'photo/' + fake_A_name)
        save_image(fake_B, conf.output_dir + 'seg/' + fake_B_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='models/')
    parser.add_argument('--output_dir', type=str, default='output1/')
    parser.add_argument('--test', action='store_true', help='test flag')
    opt = parser.parse_args()
    main(opt)
