import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import ImageOps
def get_data_loaders(batch_size=128, valid_size=0.2):
    train_dir = './train_images'
    test_dir = './test_images'

    transform = transforms.Compose(
        [transforms.Grayscale(), 
         transforms.ToTensor(), 
         transforms.Normalize(mean=(0.5,),std=(0.5,))])

    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    num_train = len(train_data)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size * num_train))
    train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

    train_sampler = SubsetRandomSampler(train_new_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, valid_loader, test_loader