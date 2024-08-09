#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def set_learning_rate(model, init_lr, backbone_lr=0.1, finetune_layer='fc'):
    finetune_layer += '.'
    backbone = []
    finetune = []

    print("\nSet learning rate -------------------------------")
    print('1. Classification layers: {}:'.format(init_lr))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if finetune_layer in name:
            finetune.append(param)
            print("   ", name)
        else:
            backbone.append(param)
    print('2. Backbone layers      : {}:'.format(init_lr * backbone_lr))

    return [
        {'params': finetune, 'lr': init_lr},
        {'params': backbone, 'lr': init_lr * backbone_lr}
    ]
