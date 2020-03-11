import torch
from torchvision import transforms


def get_transform(train):
    trans = []
    trans.append(transforms.ToTensor())

    if train:
        # transforms.append(transforms.RandomHorizontalFlip(0.5))
        pass
    return transforms.Compose(trans)


def collate_fn(batch):
    return tuple(zip(*batch))



