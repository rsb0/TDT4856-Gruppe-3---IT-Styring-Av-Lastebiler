import os
import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from engine import train_one_epoch, evaluate
from PIL import Image
from pycocotools.coco import COCO

from utils_custom import get_transform, collate_fn_custom, get_model_instance_detection

data_dir = 'data/images'
coco_dir = 'coco/output.json'


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
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

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


def train_func():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = fuelDataset(
        root='data/images', annotations='coco/output.json', transforms=get_transform(train=False)
    )
    dataset_test = fuelDataset(
        root='data/images', annotations='coco/output.json', transforms=get_transform(train=False)
    )

    # print('Length ', len(dataset))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-5])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn_custom
    )

    print('hyyyyyy')

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn_custom
    )

    print('haaaaaaa')

    # get the model using our helper function
    model = get_model_instance_detection(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 4

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == "__main__":

    print('Ey ey!')
    """

    data_dir = 'data/images'
    coco_dir = 'coco/output.json'

    print('Ey')

    my_dataset = fuelDataset(
        root=data_dir, annotations=coco_dir, transforms=get_transform(train=False)
    )

    print('Oh')

    train_batch_size = 2

    my_dataloader = torch.utils.data.DataLoader(
        my_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_custom
    )

    print('Hey!')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Dataloader is iterable over dataset
    for imgs, annotations in my_dataloader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        print(annotations)

    print('Hihihi')

    # 2 classes; Only target class or background
    num_classes = 2
    num_epochs = 10
    # model = get_model_instance_detection(num_classes)

    # For training
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    print('Has loaded model')

    images,targets = next(iter(my_dataloader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    print('Has loaded data from dataloader.')

    output = model(images,targets)   # Returns losses and detections
    print('Has finished running forward pass.')

    # For inference
    model.eval()
    print('Has evaluated.')
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)           # Returns predictions
    """
    train_func()