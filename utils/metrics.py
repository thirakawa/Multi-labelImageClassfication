#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
from sklearn import metrics


def AP(output, target):
    """Average Precision
    This function is copied from https://github.com/tk1980/TwoWayMultiLabelLoss
    """

    if len(target) == 0 or np.all(target==0):
        return -1

    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ > 0
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


# mean Average precision
def mAP(targs, preds):
    """ mean Average Precision
    This function is copied from https://github.com/tk1980/TwoWayMultiLabelLoss
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        nzidx = targets >= 0
        # compute average precision
        ap[k] = AP(scores[nzidx], targets[nzidx])
    return 100 * ap[ap>=0].mean(), 100 * ap


class MultilabelMetrics(object):
    """MultilabelMetrics
    
    Calculates several evaluation metrices for multi-label classification:
        * Accuracy, Precision, Recall, F1-score, and AP
        * Mean the above metrices over classes

    NOTE:
        The shape of confusion matrix:
            y-axis (vertical): true
            x-axis (horizontal): predicted
        I've referenced the following web-site: https://takake-blog.com/python-confusion-matrix/
    """

    def __init__(self, num_attributes, attr_name_list=None, verbose=False):
        self.num_attributes = num_attributes

        if attr_name_list is not None:
            assert len(attr_name_list) == self.num_attributes, "the number of attribute names (%d) is different with num_attributes (%d)" % (len(attr_name_list), num_attributes)
            self.attr_name_list = attr_name_list
        else:
            self.attr_name_list = list(range(self.num_attributes))

        ### stacked predicted score (for AP)
        self.pred_score_list = []
        self.pred_label_list = []
        self.true_label_list = []

    def stack(self, true_label, pred_score):

        ### convert torch.Tensor --> numpy.array
        if isinstance(true_label, torch.Tensor):
            true_label = true_label.cpu().data.numpy()
        if isinstance(pred_score, torch.Tensor):
            pred_score = pred_score.cpu().data.numpy()
        
        ### stack results
        self.pred_score_list.append(pred_score)
        self.true_label_list.append(true_label)

        ### binarize label_preds
        pred_label = pred_score.copy()
        pred_label[pred_label >= 0.5] = 1.0
        pred_label[pred_label < 0.5] = 0.0

        self.pred_label_list.append(pred_label)

    def get_score(self):

        ### concat stacked arrays
        _true_label_arr = np.concatenate(self.true_label_list, axis=0).astype(np.int32)
        _pred_label_arr = np.concatenate(self.pred_label_list, axis=0).astype(np.int32)
        _pred_score_arr = np.concatenate(self.pred_score_list, axis=0).astype(np.float32)

        ### calc. metrics
        class_pre = metrics.precision_score(y_true=_true_label_arr, y_pred=_pred_label_arr, average=None)
        mean_pre = np.mean(class_pre)

        class_rec = metrics.recall_score(y_true=_true_label_arr, y_pred=_pred_label_arr, average=None)
        mean_rec = np.mean(class_rec)

        class_f1  = metrics.f1_score(y_true=_true_label_arr, y_pred=_pred_label_arr, average=None)
        mean_f1  = np.mean(class_f1)

        ### TODO: which is better ???
        class_ap  = metrics.average_precision_score(y_true=_true_label_arr, y_score=_pred_score_arr, average=None)
        mean_ap  = np.mean(class_ap)
        # mean_ap, class_ap = mAP(targs=_true_label_arr, preds=_pred_score_arr)

        dict_pre, dict_rec, dict_f1, dict_ap = {}, {}, {}, {}
        for i in range(self.num_attributes):
            dict_pre[self.attr_name_list[i]] = float(class_pre[i])
            dict_rec[self.attr_name_list[i]] = float(class_rec[i])
            dict_f1[self.attr_name_list[i]] = float(class_f1[i])
            dict_ap[self.attr_name_list[i]] = float(class_ap[i])

        dict_pre['mean'] = float(mean_pre)
        dict_rec['mean'] = float(mean_rec)
        dict_f1['mean'] = float(mean_f1)
        dict_ap['mean'] = float(mean_ap)

        return {'precision': dict_pre, 'recall': dict_rec, 'f1-score': dict_f1, 'average_precision': dict_ap}

    def clear(self):
        """Clear stacked results"""
        self.pred_score_list = []
        self.pred_label_list = []
        self.true_label_list = []