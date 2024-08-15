#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os.path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset


NUS_NUM_CLASSES = 81

### NOTE: tuple of 81 attribute names of NUS-WIDE
#   this attribute order is based on Concepts81.txt.
#   Please do NOT change.
NUS_ATTRIBUTE_NAMES = (
    'airport', 'animal', 'beach', 'bear', 'birds',
    'boats', 'book', 'bridge', 'buildings', 'cars',
    'castle', 'cat', 'cityscape', 'clouds', 'computer',
    'coral', 'cow', 'dancing', 'dog', 'earthquake',
    'elk', 'fire', 'fish', 'flags', 'flowers',
    'food', 'fox', 'frost', 'garden', 'glacier',
    'grass', 'harbor', 'horses', 'house', 'lake',
    'leaf', 'map', 'military', 'moon', 'mountain',
    'nighttime', 'ocean', 'person', 'plane', 'plants',
    'police', 'protest', 'railroad', 'rainbow', 'reflection',
    'road', 'rocks', 'running', 'sand', 'sign',
    'sky', 'snow', 'soccer', 'sports', 'statue',
    'street', 'sun', 'sunset', 'surf', 'swimmers',
    'tattoo', 'temple', 'tiger', 'tower', 'town',
    'toy', 'train', 'tree', 'valley', 'vehicle',
    'water', 'waterfall', 'wedding', 'whales', 'window',
    'zebra'
)


### NOTE: ImageNet statistics
#   This will be used for normalization of NUS-WIDE images.
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
NUS_TRANS_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    Lighting(0.1),
    transforms.Normalize(mean=IMAGENET_STATS['mean'], std=IMAGENET_STATS['std'])
])
NUS_TRANS_TEST = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_STATS['mean'], std=IMAGENET_STATS['std'])
])


class NUSWIDEDataset(torch.utils.data.Dataset):

    def __init__(self, root, split, transform=None, target_transform=None):
        super().__init__()

        self.root = os.path.join(root, 'NUS-WIDE')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # train or test?
        if split == 'train':
            _img_list_name = os.path.join(self.root, 'ImageList/TrainImagelist.txt')
        else:
            _img_list_name = os.path.join(self.root, 'ImageList/TestImagelist.txt')

        # read image list file
        with open(_img_list_name, 'r') as f:
            self.data = f.readlines()
            self.data = list(map(str.strip, self.data))

        # read labels
        self.label = self._load_label_from_text()

    def _load_label_from_text(self):
        label = torch.zeros([len(self.data), NUS_NUM_CLASSES], dtype=torch.float32)

        for _i, _an in enumerate(NUS_ATTRIBUTE_NAMES):
            if self.split =='train':
                _label_file_name = os.path.join(self.root, 'Groundtruth/TrainTestLabels/Labels_%s_Train.txt' % _an)
            else:
                _label_file_name = os.path.join(self.root, 'Groundtruth/TrainTestLabels/Labels_%s_Test.txt' % _an)
            with open(_label_file_name, 'r') as f:
                _label_tmp = torch.tensor(list(map(int, f.readlines())))
            label[:, _i] = _label_tmp

        return label

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.root, 'Images', self.data[item].replace('\\', '/')))
        label = self.label[item, :]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.data)


def load_nus_wide_dataset(data_root, batch_size, num_workers):

    train_dataset = NUSWIDEDataset(root=data_root, split='train', transform=NUS_TRANS_TRAIN)
    test_dataset = NUSWIDEDataset(root=data_root, split='test', transform=NUS_TRANS_TEST)

    kwargs = {'num_workers': num_workers, 'pin_memory': False}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_dataset, test_dataset, train_loader, test_loader


if __name__ == '__main__':

    train_data, test_data, train_loader, test_loader = load_nus_wide_dataset("../data", batch_size=128, num_workers=1)

    for image, label in train_loader:
        print(image.size(), label.size())
        break
