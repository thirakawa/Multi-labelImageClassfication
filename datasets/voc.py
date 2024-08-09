#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import os
from xml.etree.ElementTree import Element as ET_Element

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse


import torch
from torchvision.datasets import VOCDetection
from torchvision import transforms
from torch.utils.data import DataLoader

from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image



VOC_NUM_CLASSES = 20


### NOTE: tuple of 20 category names of Pascal VOC dataset
VOC_ATTRIBUTE_NAMES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
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
#   We follow the same settings as ??????
VOC_TRANS_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    Lighting(0.1),
    transforms.Normalize(mean=IMAGENET_STATS['mean'], std=IMAGENET_STATS['std'])
])
VOC_TRANS_VAL = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_STATS['mean'], std=IMAGENET_STATS['std'])
])


class VOCClassification(VOCDetection):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Any:
        children = list(node)
        target = torch.zeros(VOC_NUM_CLASSES, dtype=torch.float)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                for _d in def_dic["object"]:
                    target[VOC_ATTRIBUTE_NAMES.index(_d['name'])] = 1
                def_dic["object"] = [def_dic["object"]]
        return target


### NOTE: function to load Pascal VOC dataset and data loader
def load_voc_dataset(data_root, year, batch_size, num_workers):
    # TODO: add test set for 2007
    train_dataset = VOCClassification(root=data_root, year=year,image_set='train',
                                      download=False, transform=VOC_TRANS_TRAIN)
    val_dataset   = VOCClassification(root=data_root, year=year, image_set='val',
                                      download=False, transform=VOC_TRANS_VAL)

    kwargs = {'num_workers': num_workers, 'pin_memory': False}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_dataset, val_dataset, train_loader, val_loader


if __name__ == '__main__':

    train_data, val_data, train_loader, val_loader = load_voc_dataset(data_root="../data", year='2007', batch_size=128, num_workers=1)

    for image, label in train_loader:
        print(image.size(), label.size())
        break
