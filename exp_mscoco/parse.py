#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
from argparse import ArgumentParser

from models import MODEL_NAMES
from losses import LOSS_NAMES


def argument_parser():
    _arg_parser = ArgumentParser(add_help=True)

    ### mode settings (train or test?)
    _arg_parser.add_argument('--mode', type=str, default='train', choices=('train', 'test'), help='train or test?')

    ### network settings
    _arg_parser.add_argument('--model', type=str, default='resnet50', choices=MODEL_NAMES, help='network model')
    _arg_parser.add_argument('--pretrained', action='store_true', help='use pretrained network model as initial parameter')

    ### loss settings
    _arg_parser.add_argument('--loss', type=str, default='bce', choices=LOSS_NAMES, help='loss function')
    _arg_parser.add_argument('--fl_alpha', type=float, default=-1.0, help='alpha of focal loss')
    _arg_parser.add_argument('--fl_gamma', type=float, default=2.0, help='gamma of focal loss')
    _arg_parser.add_argument('--asl_gamma_neg', type=float, default=4.0, help='negative gamma of asymmetric loss')
    _arg_parser.add_argument('--asl_gamma_pos', type=float, default=0.0, help='positive gamma of asymmetric loss')
    _arg_parser.add_argument('--asl_clip', type=float, default=0.05, help='clip value of asymmetric loss')
    _arg_parser.add_argument('--tml_t_pos', type=float, default=4.0, help='t_positive of two-way multi-label loss')
    _arg_parser.add_argument('--tml_t_neg', type=float, default=1.0, help='t_negaive of two-way multi-label loss')

    ### dataset path
    _arg_parser.add_argument('--data_root', type=str, default="../data", help='path to CelebA dataset directory')

    ### traininig settings
    _arg_parser.add_argument('--logdir', type=str, required=True, help='directory for storing train log and checkpoints')
    _arg_parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size')
    _arg_parser.add_argument('--epochs', type=int, default=40, help='the number of training epochs')
    _arg_parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    _arg_parser.add_argument('--b_lr', type=float, default=0.1, help='learning rate for backbone layers')
    _arg_parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD optimizer')
    _arg_parser.add_argument('--wd', type=float, default=1e-4, help='weight decay of SGD optimizer')
    _arg_parser.add_argument('--use_nesterov', action='store_true', help='use nesterov accelerated SGD')
    _arg_parser.add_argument('--scheduler', type='str', default='cos', choices=['cos', 'onecycle'], help='LR scheduler type')
    _arg_parser.add_argument('--num_workers', type=int, default=5, help='the number of multiprocess workers for data loader')

    ### resume settings
    _arg_parser.add_argument('--resume', type=str, default=None, help='filename of checkpoint for resuming the training')

    ### random seed settings
    _arg_parser.add_argument('--seed', type=int, default=None, help='random seed')

    ### GPU settings
    _arg_parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    args = _arg_parser.parse_args()

    ### set random seed if it is None
    if args.seed is None:
        args.seed = random.randint(0, 10000)
        print("Set random seed as %d ..." % args.seed)

    return args
