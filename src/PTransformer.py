import torchvision.transforms as tfs

# Auxiliary torch.transforms object that convert img to tensor
directly_tfs = tfs.Compose([
    tfs.ToTensor()
])

directly_tfs_gray = tfs.Compose([
    tfs.Grayscale(),
    tfs.ToTensor()
])

tfs_32 = tfs.Compose([
    tfs.Resize(32),
    tfs.ToTensor()
])

tfs_32_gray = tfs.Compose([
    tfs.Grayscale(),
    tfs.Resize(32),
    tfs.ToTensor()
])

tfs_28 = tfs.Compose([
    tfs.Resize(28),
    tfs.ToTensor()
])

tfs_28_gray = tfs.Compose([
    tfs.Grayscale(),
    tfs.Resize(28),
    tfs.ToTensor()
])

tfs_224 = tfs.Compose([
    tfs.Resize((224, 224)),
    tfs.ToTensor()
])

tfs_227 = tfs.Compose([
    tfs.Resize((227, 227)),
    tfs.ToTensor()
])