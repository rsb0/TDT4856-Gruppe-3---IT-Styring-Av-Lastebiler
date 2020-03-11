import os
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO


class fuelDataset(data.Dataset):

    def __init__(self, root, annotations, transforms=None):
        self.root = root
        self.transforms = transforms
        
        # self.imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))

        self.coco = COCO(annotations)
        self.ids = list(sorted(self.coco.img.keys()))

    def __getitem__(self, idx):
        
