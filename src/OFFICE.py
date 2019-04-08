import numpy as np
from torch.utils import data
from torchvision import datasets, transforms

# For Office31 datasets data_loader
def get_office31_dataloader(root, case, batch_size):
    print('[INFO] Loading datasets: {}'.format(case))
    datas = {
        'amazon_train': root + "train/amazon/",
        'amazon_test': root + 'test/amazon/',
        'dslr_train': root + 'train/dslr/',
        'dslr_test': root + 'test/dslr/',
        'webcam_train': root + 'train/webcam',
        'webcam_test': root + 'test/webcam',
        'amazon': 'dataset/office31/amazon/images/',
        'dslr': 'dataset/office31/dslr/images/',
        'webcam': 'dataset/office31/webcam/images/'
    }
    means = {
        'amazon': [0.79235075407833078, 0.78620633471295642, 0.78417965306916637],
        'webcam': [0.61197983011509638, 0.61876474000372972, 0.61729662103473015],
        'dslr': [],
        'imagenet': [0.485, 0.456, 0.406]
    }
    stds = {
        'amazon': [0.27691643643313618, 0.28152348841965347, 0.28287296762830788],
        'webcam': [0.22763857108616978, 0.23339382150450594, 0.23722725519031848],
        'dslr': [],
        'imagenet': [0.229, 0.224, 0.225]
    }

    img_size = (224, 224)

    transform = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # transforms.Normalize(means['imagenet'], stds['imagenet']),
    ]

    data_loader = data.DataLoader(
        dataset=datasets.ImageFolder(
            datas[case],
            transform=transforms.Compose(transform)
        ),
        num_workers=2,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return data_loader