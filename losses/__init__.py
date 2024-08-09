#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .binaryfocalloss import BinaryFocalLossWithLogits, WeightedBinaryFocalLossWithLogits
from .twowayloss import TwoWayLoss
from .asymmetryloss import AsymmetricLoss, AsymmetricLossOptimized, ASLSingleLabel


__all__ = [
    'BinaryFocalLossWithLogits', 'WeightedBinaryFocalLossWithLogits', 'TwoWayLoss', 'AsymmetricLoss', 'AsymmetricLossOptimized', 'ASLSingleLabel'
]


LOSS_NAMES = ['bce', 'fl', 'asl', 'tml']


def load_loss_function(loss_name, args):
    import torch.nn as nn

    if loss_name not in LOSS_NAMES:
        print("ERROR: invalid loss name: %s" % loss_name)
        exit(-1)

    print('\nSet loss function -------------------------------')
    print("  Loss: %s" % loss_name)
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'fl':
        print("    params: alpha=%f, gamma=%f" % (args.fl_alpha, args.fl_gamma))
        return BinaryFocalLossWithLogits(alpha=args.fl_alpha, gamma=args.fl_gamma)
    elif loss_name == 'asl':
        print("    params: gamma_neg=%f, gamma_pos=%f, clip=%f" % (args.asl_gamma_neg, args.asl_gamma_pos, args.asl_clip))
        return AsymmetricLossOptimized(gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos, clip=args.asl_clip)
    elif loss_name == 'tml':
        print("    params: Tp=%f, Tn=%f" % (args.tml_t_pos, args.tml_t_neg))
        return TwoWayLoss(Tp=args.tml_t_pos, Tn=args.tml_t_neg)
