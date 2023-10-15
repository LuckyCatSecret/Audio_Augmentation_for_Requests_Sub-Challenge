import os
import torch
import torch.distributed as dist
import numpy as np
from scipy import interpolate
from sklearn import metrics


def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))
    predict1 = np.argmax(output,axis = 1)
    predict1 = (np.arange(classes_num) == predict1[:,None]).astype(np.int64)
    mUAR = metrics.recall_score(target,predict1,average='macro')


    # Class-wise statistics
    for k in range(classes_num):

        # Average precision

        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        # auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)
        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])


        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': 0,
                'UAR':mUAR,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc
                }
        stats.append(dict)

    return stats