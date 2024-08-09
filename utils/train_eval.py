#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import torch
from torch.cuda.amp import GradScaler, autocast
from .metrics import MultilabelMetrics


def training(model, data_loader, optimizer, criterion, writer, iteration, loss_sum, log_step=100):
    _use_amp = True
    scaler = GradScaler(enabled=_use_amp)

    model.train()
    for image, label in data_loader:
        iteration += 1
        image, label = image.cuda(), label.to(torch.float32).cuda()
        model.zero_grad()

        with autocast(enabled=_use_amp):
            output = model(image)
            loss = criterion(output, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()

        if iteration % log_step == 0:
            print("iteration: %06d loss: %0.8f" % (iteration, loss_sum / log_step))
            writer.add_scalar("00_loss/loss", loss_sum / log_step, iteration)
            loss_sum = 0.0

    return iteration, loss_sum


def evaluation(model, data_loader, attribute_names, writer=None, epoch=None, result_dir_path=None):
    metrics = MultilabelMetrics(num_attributes=len(attribute_names), attr_name_list=attribute_names)

    model.eval()
    with torch.no_grad():
        for image, label in data_loader:
            image = image.cuda()
            output = model(image)

            # binarize output
            label = label.data
            metrics.stack(true_label=label, pred_score=torch.sigmoid(output))

    ### compute & print accuracy
    score = metrics.get_score()
    print("\nScore ------------------")
    print("    mean Precision:", score['precision']['mean'])
    print("    mean Recall   :", score['recall']['mean'])
    print("    mean F1-score :", score['f1-score']['mean'])
    print("    mean AP       :", score['average_precision']['mean'], "\n")

    ### write TensorBoard
    if writer is not None and epoch is not None:
        _keys = list(score['precision'].keys())
        for k in _keys:
            if _keys == 'mean': continue
            writer.add_scalar('02_avg_precision/%s' % str(k), score['average_precision'][k], epoch)
            writer.add_scalar('03_f1-score/%s' % str(k),      score['f1-score'][k], epoch)
            writer.add_scalar('05_precision/%s' % str(k),     score['precision'][k], epoch)
            writer.add_scalar('06_recall/%s' % str(k),        score['recall'][k], epoch)

        writer.add_scalar('01_mean_score/precision',     score['precision']['mean'], epoch)
        writer.add_scalar('01_mean_score/recall',        score['recall']['mean'], epoch)
        writer.add_scalar('01_mean_score/f1-score',      score['f1-score']['mean'], epoch)
        writer.add_scalar('01_mean_score/avg_precision', score['average_precision']['mean'], epoch)

    ### save as json
    if result_dir_path is not None:
        with open(os.path.join(result_dir_path, "score.json"), 'w') as f:
            json.dump(score, f, indent=4)

    return score
