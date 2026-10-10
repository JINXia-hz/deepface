import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image


def _read_image(path):
    with Image.open(path) as im:
        return np.asarray(im, dtype=np.uint8)


class InMemoryImageFolder(Dataset):
    """Preload every image of an ImageFolder directory into one uint8 tensor.

    Items are returned normalized to [-1, 1] float tensors, equivalent to
    Grayscale + ToTensor + Normalize(0.5, 0.5).
    """

    def __init__(self, root):
        folder = torchvision.datasets.ImageFolder(root)
        self.root = root
        self.paths = [p for p, _ in folder.samples]
        self.targets = torch.tensor(folder.targets, dtype=torch.long)
        self.images = None

    def _load(self):
        print(f"Preloading {len(self.paths)} images from {self.root} ...")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            arrays = list(pool.map(_read_image, self.paths))
        h, w = arrays[0].shape
        self.images = torch.from_numpy(np.stack(arrays)).view(-1, 1, h, w)
        print(f"Preloaded {self.root}: {tuple(self.images.shape)}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.images is None:
            self._load()
        x = self.images[idx].float().mul_(2.0 / 255.0).sub_(1.0)
        return x, self.targets[idx]


def get_data_loaders(batch_size=256, valid_size=0.2):
    train_dir = './train_images'
    test_dir = './test_images'

    train_data = InMemoryImageFolder(train_dir)
    test_data = InMemoryImageFolder(test_dir)

    num_train = len(train_data)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size * num_train))
    train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

    train_sampler = SubsetRandomSampler(train_new_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, valid_loader, test_loader
