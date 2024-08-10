#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
sys.path.append("../")

import os
import re
import random

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

### import from multilabel module
from datasets.celeba import load_celeba_dataset, CELEBA_NUM_CLASSES, CELEBA_ATTRIBUTE_NAMES
from models import load_network_model
from losses import load_loss_function
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.args import save_args
from utils.train_eval import training, evaluation

### import from same directory
from parse import argument_parser


CHECKPOINT_STEP = 1
RESULT_DIR_TRAIN = "result_%s_train"
RESULT_DIR_VAL   = "result_%s_val"
RESULT_DIR_TEST  = "result_%s_test"


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def main():
    args = argument_parser()

    ### GPU (device) settings -----------------------------
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    print("use cuda:", use_cuda)
    assert use_cuda, "This training script needs CUDA (GPU)."

    ### set random seed -----------------------------------
    fix_random_seed(args.seed)

    ### dataset -------------------------------------------
    print("load dataset")
    train_dataset, _, _, train_loader, val_loader, test_loader = load_celeba_dataset(args.data_root, args.batch_size, args.num_workers)

    ### network model -------------------------------------
    model = load_network_model(
        model_name=args.model,
        num_classes=CELEBA_NUM_CLASSES,
        pretrained=args.pretrained
    )

    ### optimizer -----------------------------------------
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.use_nesterov)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)

    ### loss function -------------------------------------
    criterion = load_loss_function(args.loss, args)

    ### CPU or GPU (Data Parallel) ------------------------
    model = nn.DataParallel(model).cuda()
    criterion = criterion.cuda()
    cudnn.benchmark = True

    # -------------------------------------------------------------------------
    # MODE: evaluation
    # -------------------------------------------------------------------------
    # resume ----------------------------------------------
    if args.resume is not None:
        print("Load checkpoint for resuming a training ...")
        print("    checkpoint:", os.path.join(args.logdir, args.resume))
        model, optimizer, scheduler, _, _, _ = load_checkpoint(os.path.join(args.logdir, args.resume), model, optimizer, scheduler)
        resume_postfix = re.match(r'checkpoint-([a-zA-Z0-9_]+)\.pt', args.resume).group(1)

    if args.mode == 'test':
        print("mode: evaluation\n")

        ### validation data ---------------------
        print("evaluate test data ...")
        evaluation(
            model=model, data_loader=val_loader,
            attribute_names=CELEBA_ATTRIBUTE_NAMES, writer=None,
            result_dir_name=os.path.join(args.logdir, RESULT_DIR_VAL % resume_postfix)
        )

        ### test data ---------------------------
        print("evaluate test data ...")
        evaluation(
            model=model, data_loader=test_loader,
            attribute_names=CELEBA_ATTRIBUTE_NAMES, writer=None,
            result_dir_name=os.path.join(args.logdir, RESULT_DIR_TEST % resume_postfix)
        )

        print("evaluation; done.\n")
        exit(0)

    # -------------------------------------------------------------------------
    # MODE: training
    # -------------------------------------------------------------------------
    # tensorboard -----------------------------------------
    writer = SummaryWriter(log_dir=args.logdir)
    log_dir = writer.file_writer.get_logdir()
    save_args(os.path.join(log_dir, 'args.json'), args)

    ### logging variables ---------------------------------
    initial_epoch, iteration = 1, 0
    best_score = 0.0
    loss_sum = 0.0

    for epoch in range(initial_epoch, args.epochs + 1):

        ### train
        print("epoch:", epoch)
        iteration, loss_sum = training(model, train_loader, optimizer, criterion, writer, iteration, loss_sum)

        ### validation
        print("evaluation ...")
        score = evaluation(model, val_loader, CELEBA_ATTRIBUTE_NAMES, writer, epoch, result_dir_path=None)

         ### update learning rate
        scheduler.step()

        ### save model
        print("save model ...")
        ### 1. best validation accuracy
        if best_score < score['average_precision']['mean']:
            best_score = score['average_precision']['mean']
            save_checkpoint(os.path.join(log_dir, "checkpoint-best.pt"), model, optimizer, scheduler, best_score, epoch, iteration)

        ### 2. at regular intervals
        if epoch % CHECKPOINT_STEP == 0:
            save_checkpoint(os.path.join(log_dir, "checkpoint-%04d.pt" % epoch), model, optimizer, scheduler, best_score, epoch, iteration)

        print("epoch:", epoch, "; done.\n")

    ### save final model & close tensorboard writer -------
    print("save final model")
    save_checkpoint(os.path.join(log_dir, "checkpoint-final.pt"), model, optimizer, scheduler, best_score, epoch, iteration)
    writer.close()
    print("training; done.")


if __name__ == '__main__':
    main()
