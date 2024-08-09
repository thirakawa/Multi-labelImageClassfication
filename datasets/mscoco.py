#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os.path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset


COCO_NUM_CLASSES = 80


### NOTE: tuple of 80 category names of MSCOCO
#   This information is refered from https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/.
#   Please do NOT change.
COCO_ATTRIBUTE_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
)


### NOTE: ImageNet statistics
#   This will be used for normalization of COCO images.
#   Because the experimental settings of Twoway Multilabel Loss (Kobayashi, CVPR2023) uses the pretrained weight by ImageNet as initial network params.
IMAGENET_STATS = {
    'mean': [0.485, 0.456, 0.406],
     'std': [0.229, 0.224, 0.225]
}
IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd=0.1, 
                 eigval=IMAGENET_PCA['eigval'], 
                 eigvec=IMAGENET_PCA['eigvec']):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        # Create a random vector of 3x1 in the same type as img
        alpha = img.new_empty(3,1).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .matmul(alpha * self.eigval.view(3, 1))

        return img.add(rgb.view(3, 1, 1).expand_as(img))


### NOTE: transforms for image data
#   We follow the same settings as those developed by T. Kobayashi (CVPR 2023).
COCO_TRANS_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    Lighting(0.1),
    transforms.Normalize(mean=IMAGENET_STATS['mean'], std=IMAGENET_STATS['std'])
])
COCO_TRANS_VAL = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_STATS['mean'], std=IMAGENET_STATS['std'])
])


class CocoClassification(VisionDataset):

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # for _k in self.cat2cat.keys():
        #     print(_k, ":", self.cat2cat[_k])

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        output = torch.zeros(80, dtype=torch.float)
        for obj in target:
            output[self.cat2cat[obj['category_id']]] = 1
        target = output

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self) -> int:
        return len(self.ids)


### NOTE: function to load MSCOCO dataset and data loader
def load_coco_dataset(data_root, batch_size, num_workers):
    train_dataset = CocoClassification(root=os.path.join(data_root, "MSCOCO/images/train2014"),
                                       annFile=os.path.join(data_root, "MSCOCO/annotations/instances_train2014.json"),
                                       transform=COCO_TRANS_TRAIN)
    val_dataset   = CocoClassification(root=os.path.join(data_root, "MSCOCO/images/val2014"),
                                       annFile=os.path.join(data_root, "MSCOCO/annotations/instances_val2014.json"),
                                       transform=COCO_TRANS_VAL)

    kwargs = {'num_workers': num_workers, 'pin_memory': False}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_dataset, val_dataset, train_loader, val_loader


if __name__ == '__main__':

    train_data, val_data, train_loader, val_loader = load_coco_dataset(data_root="../data", batch_size=512, num_workers=1)

    print("the number of len of train loader", len(train_loader))

    for image, label in train_loader:
        print(image.size(), label.size())
        break
