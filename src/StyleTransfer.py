import torch
import os
import torch.nn.functional as F
import torchvision.transforms as tfs
from torch import nn, optim
from torchvision.models import vgg19
from torch.autograd import Variable
from model.MVGG import MultiVGG19
import matplotlib.pyplot as plt
from PIL import Image

# pre and post processing for images
img_size = 512
prep = tfs.Compose([tfs.Resize(img_size),
                    tfs.ToTensor(),
                    tfs.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                    tfs.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]),
                    tfs.Lambda(lambda x: x.mul_(255)),
                    ])
postpa = tfs.Compose([tfs.Lambda(lambda x: x.mul_(1./255)),
                      tfs.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1, 1, 1]),
                      tfs.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                      ])
postpb = tfs.Compose([tfs.ToPILImage()])


def img_clip(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img

def main():
    pivot_vgg = vgg19(pretrained=True)
    tgt_model = MultiVGG19(pivot_vgg)
    tgt_model.batch_required_grad(False)

    if torch.cuda.is_available():
        tgt_model = tgt_model.cuda()

    style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    content_layers = ['r42']

    data_path = os.getcwd() + "/../data/style_transfer/"
    style_image = data_path + "style_img/" + "vangogh_starry_night.jpg"
    content_image = data_path + "content_img/" + "Tuebingen_Neckarfront.jpg"
    output_image = data_path + "output/" + "test.jpg"
    imgs = [Image.open(style_image), Image.open(content_image)]

    imgs_torch = [prep(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
    style_image, content_image = imgs_torch

    # Initialize the 'white noisy' image by clone current content img
    #opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True)
    opt_img = Variable(content_image.data.clone(), requires_grad=True)

    # Pre-defined weight vector from the paper
    style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
    #style_weights = [1e3 / n ** 2 for n in [32, 64, 128, 256, 256]]
    content_weights = [1e0]

    # compute optimization targets
    style_targets = [A.detach() for A in tgt_model(style_image, style_layers, gram_flag=True)]
    content_targets = [A.detach() for A in tgt_model(content_image, content_layers)]

    # optimizer = optim.LBFGS([opt_img])
    optimizer = optim.Adam([opt_img])
    criterion = nn.MSELoss(reduction='sum')
    max_iter = 500
    show_iter = 50
    n_iter = [0]

    # while n_iter[0] <= max_iter:
    #
    #     def closure():
    #         optimizer.zero_grad()
    #         # Forward once
    #         # out = tgt_model(opt_img, loss_layers)
    #         style_out = tgt_model(opt_img, style_layers, gram_flag=True)
    #         content_out = tgt_model(opt_img, content_layers)
    #
    #         # Compute the loss respectively
    #         style_loss = [style_weights[idx] * criterion(x, style_targets[idx]) for idx, x in enumerate(style_out)]
    #         content_loss = [content_weights[idx] * criterion(x, content_targets[idx]) for idx, x in enumerate(content_out)]
    #         final_loss = style_loss + content_loss
    #         # layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
    #         loss = sum(final_loss)
    #         loss.backward()
    #         n_iter[0] += 1
    #         # print loss
    #         if n_iter[0] % show_iter == (show_iter - 1):
    #             print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
    #         return loss
    #
    #     optimizer.step(closure)

    while n_iter[0] <= max_iter:
        optimizer.zero_grad()
        style_out = tgt_model(opt_img, style_layers, gram_flag=True)
        content_out = tgt_model(opt_img, content_layers)
        style_loss = [style_weights[idx] * criterion(x, style_targets[idx]) for idx, x in enumerate(style_out)]
        content_loss = [content_weights[idx] * criterion(x, content_targets[idx]) for idx, x in enumerate(content_out)]
        final_loss = style_loss + content_loss
        loss = sum(final_loss)
        loss.backward()
        n_iter[0] += 1
        # print loss
        if n_iter[0] % show_iter == (show_iter - 1):
            print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
        
    out_img = img_clip(opt_img.data[0].cpu().squeeze())
    out_img.save(output_image)
    plt.imshow(out_img)
    plt.gcf().set_size_inches(10, 10)


if __name__ == "__main__":
    main()
