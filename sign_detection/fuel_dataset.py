import os
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO

from utils import get_transform, collate_fn


class fuelDataset(torch.utils.data.Dataset):

    def __init__(self, root, annotations, transforms=None):
        self.root = root
        self.transforms = transforms
        
        # self.imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))

        self.coco = COCO(annotations)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):

        # Coco annotations
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        coco_annotation = coco.loadAnns(ann_ids)

        # Load image
        img_path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, img_path))

        num_objs = len(coco_annotation)

        # Bounding boxes
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Labels - only one label in this case
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Tensorize image id
        img_id = torch.tensor([img_id])

        # Size of bounding box
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Target dictionary
        target = {}

        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = img_id
        target['area'] = areas
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":

    data_dir = 'data/images'
    coco_dir = 'coco/output.json'

    print('Ey')

    my_dataset = fuelDataset(
        root=data_dir, annotations=coco_dir, transforms=get_transform(train=False)
    )

    print('Oh')

    train_batch_size = 1

    my_dataloader = torch.utils.data.DataLoader(
        my_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn
    )

    print('Hey!')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Dataloader is iterable over dataset
    for imgs, annotations in my_dataloader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        print(annotations)