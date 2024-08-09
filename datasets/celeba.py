#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA


CELEBA_NUM_CLASSES = 40


### NOTE: tuple of 40 attribute names of CelebA
#   this attribute order is based on list_attr_celeba.txt.
#   Please do NOT change.
CELEBA_ATTRIBUTE_NAMES = (
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young'
)


### NOTE: MEAN and STD of CelebA
#   mean and std of RGB values on train dataset (order: R, G, B)
#   We assume that RGB value range is [0, 1]
CELEBA_TRAIN_RGB_MEAN = (0.5063454195744012, 0.42580961977997583, 0.38318672615935173)
CELEBA_TRAIN_RGB_STD  = (0.310506447934692, 0.2903443482746604, 0.2896806573348839)


### NOTE: Prior distribution of attributes on training set
#   this attribute order is based on list_attr_celeba.txt.
#   Please do NOT change.
CELEBA_FREQUENCY_HISTOGRAM = [
    0.0123710623010993, 0.02945452183485031, 0.056899264454841614, 0.022649994120001793, 0.0025270262267440557,
    0.01680033467710018, 0.026687927544116974, 0.026094455271959305, 0.02647898718714714, 0.01651584729552269,
    0.005691083613783121, 0.02259010262787342, 0.0159162487834692, 0.006390048190951347, 0.005152737721800804,
    0.007160474546253681, 0.00703524611890316, 0.004693340510129929, 0.04257423058152199, 0.050121963024139404,
    0.046457670629024506, 0.05341669172048569, 0.00452047074213624, 0.012842030264437199, 0.0924096629023552,
    0.03137582540512085, 0.004767524544149637, 0.030521685257554054, 0.008874878287315369, 0.007163196802139282,
    0.006231470964848995, 0.05314037203788757, 0.023103946819901466, 0.03537836670875549, 0.0206640362739563,
    0.005471253301948309, 0.052022166550159454, 0.01345115713775158, 0.008092200383543968, 0.08629049360752106
]


### NOTE: transforms for image data
#   We do not know optimal transforms for obtaining clear attention maps...
CELEBA_TRANS_TRAIN = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=CELEBA_TRAIN_RGB_MEAN, std=CELEBA_TRAIN_RGB_STD)
])
CELEBA_TRANS_EVAL = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CELEBA_TRAIN_RGB_MEAN, std=CELEBA_TRAIN_RGB_STD)
])


### NOTE: Frequency histogram of attribute labels on training set
def get_celeba_frequency_histogram(dataset, normalize=True):
    _hist = torch.sum(dataset.attr, dim=0)
    if normalize:
        return _hist / torch.sum(_hist)
    else:
        return _hist


### NOTE: CelebA Dataset Class
#   We use torchvision.datasets.CelebA and do not implement Dataset class by ourselves.
#   Therefore, please see official document and source code for more details.


### NOTE: function to load CelebA dataset and data loader
def load_celeba_dataset(data_root, batch_size, num_workers):
    train_dataset = CelebA(root=data_root, split='train', target_type='attr', transform=CELEBA_TRANS_TRAIN, download=False)
    val_dataset   = CelebA(root=data_root, split='valid', target_type='attr', transform=CELEBA_TRANS_EVAL, download=False)
    test_dataset  = CelebA(root=data_root, split='test', target_type='attr', transform=CELEBA_TRANS_EVAL, download=False)

    kwargs = {'num_workers': num_workers, 'pin_memory': False}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
