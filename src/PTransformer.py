import torchvision.transforms as tfs

# Auxiliary torch.transforms object that convert img to tensor
directly_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize((.5,), (.5, ))
])

directly_tfs_gray = tfs.Compose([
    tfs.Grayscale(),
    tfs.ToTensor(),
    tfs.Normalize((.5,), (.5, ))
])

tfs_32 = tfs.Compose([
    tfs.Resize(32),
    tfs.ToTensor(),
    tfs.Normalize((.5,), (.5, ))
])

tfs_32_gray = tfs.Compose([
    tfs.Grayscale(),
    tfs.Resize(32),
    tfs.ToTensor(),
    tfs.Normalize((.5,), (.5, ))
])

tfs_28 = tfs.Compose([
    tfs.Resize(28),
    tfs.ToTensor(),
    tfs.Normalize((.5,), (.5, ))
])

tfs_28_gray = tfs.Compose([
    tfs.Grayscale(),
    tfs.Resize(28),
    tfs.ToTensor(),
    tfs.Normalize((.5,), (.5, ))
])

tfs_224 = tfs.Compose([
    tfs.Resize((224, 224)),
    tfs.ToTensor()
])

tfs_227 = tfs.Compose([
    tfs.Resize((227, 227)),
    tfs.ToTensor()
])

normalize = tfs.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

tfs_224_norm = tfs.Compose([
    tfs.Resize((224, 224)),
    tfs.ToTensor(),
    normalize
])

tfs_227_norm = tfs.Compose([
    tfs.Resize((227, 227)),
    tfs.ToTensor(),
    normalize
])