#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch.nn as nn
from torchvision.models import get_model


MODEL_NAMES = [
    'resnet50',
    'resnext50_32x4d',
    'densenet169',
    'regnet_y_32gf'
]


def load_network_model(model_name, num_classes, pretrained=False):
    if model_name not in MODEL_NAMES:
        print("ERROR: invalid network model name: %s" % model_name)
        exit(-1)

    if pretrained:
        model = get_model(name=model_name, weights="DEFAULT")
        # replace FC layer
        for _l_name, _layer in model.named_modules():
            if type(_layer) is nn.Linear:
                break
        setattr(model, _l_name, nn.Linear(_layer.in_features, num_classes))
    else:
        model = get_model(name=model_name, weights=None, num_classes=num_classes)

    return model


if __name__ == '__main__':

    model1_scratch = load_network_model('resnet50', num_classes=40, pretrained=False)
    model1_ftune   = load_network_model('resnet50', num_classes=40, pretrained=True)

    model2_scratch = load_network_model('resnext50_32x4d', num_classes=40, pretrained=False)
    model2_ftune   = load_network_model('resnext50_32x4d', num_classes=40, pretrained=True)

    model3_scratch = load_network_model('densenet169', num_classes=40, pretrained=False)
    model3_ftune   = load_network_model('densenet169', num_classes=40, pretrained=True)

    model4_scratch = load_network_model('regnet_y_32gf', num_classes=40, pretrained=False)
    model4_ftune   = load_network_model('regnet_y_32gf', num_classes=40, pretrained=True)
