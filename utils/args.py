#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json


def save_args(save_filename, args):
    with open(save_filename, 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def load_args(load_filename):
    with open(load_filename, 'r') as f:
        args = json.load(f)
    return args
