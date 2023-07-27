# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config


class FullModel(nn.Module):

    def __init__(self, model, sem_loss, bd_loss):
        super(FullModel, self).__init__()
        self.model = model  # PIDNet
        self.sem_loss = sem_loss  # OhemCrossEntropy
        self.bd_loss = bd_loss  # BoundaryLoss

    def pixel_acc(self, pred, label):
        """

        Args:
            pred: (batch_size, 2, 1024, 1024)
            label: [batch_size, height, width]
                0, 255 로만 이루어져 있음.

        Returns:
        """
        _, preds = torch.max(pred, dim=1)
        # preds: (batch_size, height, width), 0 혹은 1 로만 이루어져 있음.
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, labels, bd_gt, *args, **kwargs):
        """

        Args:
            inputs: [batch_size, 3, height, width]
            labels: [batch_size, height, width]
                0, 255 로만 이루어져 있음.
            bd_gt: [batch_size, height, width]
                0, 1 로만 이루어져 있음.

        Returns:
            pixel_losses = pixel_losses[mask]
            print("pixel_losses.mean(): ", pixel_losses.mean())
        """
        # inputs: [batch_size, 3, height, width]
        # network_outputs: [batch_size, num_classes, height//8, width//8]
        network_outputs = self.model(inputs, *args, **kwargs)
        h, w = labels.size(1), labels.size(2)
        ph, pw = network_outputs[0].size(2), network_outputs[0].size(3)
        if ph != h or pw != w:
            for i in range(len(network_outputs)):
                network_outputs[i] = F.interpolate(
                    network_outputs[i],
                    size=(h, w),
                    mode='bilinear',
                    align_corners=config.MODEL.ALIGN_CORNERS)
        # [6, 2, 1024, 1024]
        x_extra_p_output = network_outputs[0]  # (batch_size, 2, height, width)
        # [6, 2, 1024, 1024]
        x_output = network_outputs[1]  # (batch_size, 2, height, width)
        # [6, 1, 1024, 1024]
        x_extra_d_output = network_outputs[2]  # (batch_size, 1, height, width)
        # S-loss (extra semantic loss) (P network)
        # 0, 255를 잘 맞추도록
        acc = self.pixel_acc(pred=x_extra_p_output, label=labels)
        loss_s = self.sem_loss(scores=[x_extra_p_output, x_output],
                               target=labels)  # OhemCrossEntropy
        # B-loss (boundary binary cross entropy loss) (D network)
        loss_b = self.bd_loss(x_extra_d_output, bd_gt)  # BoundaryLoss

        filler = torch.ones_like(labels) * config.TRAIN.IGNORE_LABEL
        bd_label = torch.where(
            F.sigmoid(x_extra_d_output[:, 0, :, :]) > 0.8, labels, filler)
        loss_sb = self.sem_loss(x_output, bd_label)  # OhemCrossEntropy
        loss = torch.unsqueeze(loss_s + loss_b + loss_sb, 0)
        return loss, network_outputs[:-1], acc, [loss_s, loss_b]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def create_logger(cfg,
                  cfg_name: str,
                  phase='train') -> Tuple[logging.Logger, str, str]:
    """

    Args:
        cfg: config
        cfg_name:
            configs/cityscapes/pidnet_large_ade.yaml
            cofings/ade/pidnet_large_ade.yaml
        phase: train or val

    Returns:
        logger
        final_output_dir:
            output/cityscapes/pidnet_large_cityscapes
            output/ade/pidnet_large_ade
        tensorboard_log_dir:
            log/cityscapes/pidnet_large_cityscapes/pidnet_large_cityscapes_time
            log/ade/pidnet_large_ade/pidnet_large_ade_time
    """
    # from pathlib import Path
    root_output_dir = Path(cfg.OUTPUT_DIR)  # "output"
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET  # cityscapes / ade
    model = cfg.MODEL.NAME  # pidnet_large /
    # cfg_name = pidnet_large_cityscapes / pidnet_large_ade
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    # output/cityscapes/pidnet_large_cityscapes
    # output/ade/pidnet_large_ade
    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_confusion_matrix(label, pred, size, num_class, ignore_label=-1):
    """Calculate the confusion matrix by given label and pred.

    Args:
        label: [batch_size, height, width]
            0, 1, 255 로만 이루어져 있음.
        pred: [batch_size, 2, height, width]
        size: (batch_size, height, width)
        num_class: 2
        ignore_label: 255

    Returns:

    """
    output = pred.cpu().numpy().transpose(0, 2, 3,
                                          1)  # (batch_size, height, width, 2)
    seg_pred = np.asarray(np.argmax(output, axis=3),
                          dtype=np.uint8)  # (batch_size, height, width)
    # size[-2], size[-1] = height, width
    seg_gt = np.asarray(label.cpu().numpy()[:, :size[-2], :size[-1]],
                        dtype=np.int)  # (batch_size, height, width)
    ignore_index = seg_gt != ignore_label  # 중요하면 1, 아니면 0
    seg_gt = seg_gt[ignore_index]  # 중요한 것만 뽑아냄
    seg_pred = seg_pred[ignore_index]  # 중요한 것만 뽑아냄

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix


def adjust_learning_rate(optimizer,
                         base_lr,
                         max_iters,
                         cur_iters,
                         power=0.9,
                         nbb_mult=10):
    lr = base_lr * ((1 - float(cur_iters) / max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr
