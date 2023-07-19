# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate

import platform
from PIL import Image

import torch

def concatenate_two_images(sv_img: Image, image):
    img = Image.fromarray(image)
    img = img.resize((sv_img.width, sv_img.height))
    img = Image.blend(img, sv_img, 0.5)
    return Image.fromarray(np.hstack((np.array(img), np.array(sv_img))))

color_map = [
(152, 251, 152),  # terrain (nature) 지역
(220, 20, 60),  # person
]

def get_torch_gpu_device(gpu_idx: int = 0) -> str:
    if IS_MAC:
        assert torch.backends.mps.is_available()
        device = f"mps:{gpu_idx}"
    else:
        assert torch.cuda.is_available()
        device = f"cuda:{gpu_idx}"
    return device


if platform.system() == "Darwin" and platform.uname().processor == "arm":
    IS_MAC = True
    device = get_torch_gpu_device()
else:
    IS_MAC = False


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
          trainloader, optimizer, full_model, writer_dict):
    """

    :param config:
    :param epoch:
    :param num_epoch:
    :param epoch_iters: 1200 / batch_size
    :param base_lr:
    :param num_iters:
    :param trainloader: torch.utils.data.DataLoader
    :param optimizer:
    :param full_model:
    :param writer_dict:
        writer_dict = {
            'writer': SummaryWriter(logdir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
    }
    """
    # Training
    full_model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    summary_writer = writer_dict['writer']  # SummaryWriter(logdir=tb_log_dir)
    global_steps = writer_dict['train_global_steps']
    for i_iter, batch in enumerate(trainloader, 0):
        """
images: 
    [batch_size, num_channels, height, width]
labels:  [batch_size, height, width]
    0, 1, 255 로만 이루어져 있음.
bd_gts: [batch_size, height, width]
    0, 1 로만 이루어져 있음.
        """
        images, labels, bd_gts, _, _ = batch
        if IS_MAC:
            images = images.to(device)
            labels = labels.long().to(device)
            bd_gts = bd_gts.float().to(device)
        else:
            images = images.cuda()  # [6, 3,
            labels = labels.long().cuda()
            bd_gts = bd_gts.float().cuda()

        losses, _, acc, loss_list = full_model(inputs=images, labels=labels, bd_gt=bd_gts)
        loss = losses.mean()
        acc = acc.mean()

        full_model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())

        lr = adjust_learning_rate(optimizer, base_lr, num_iters,
                                  i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = f"Epoch: [{epoch}/{num_epoch}] Iter:[{i_iter}/{epoch_iters}]," \
                  f" Time: {batch_time.average():.2f}, " \
                  f"lr: {[x['lr'] for x in optimizer.param_groups]}," \
                  f" Loss: {ave_loss.average():.6f}, " \
                  f"Acc: {ave_acc.average():.6f}, " \
                  f"Semantic loss: {avg_sem_loss.average():.6f}, " \
                  f"BCE loss: {avg_bce_loss.average():.6f}, " \
                  f"SB loss: {ave_loss.average() - avg_sem_loss.average() - avg_bce_loss.average():.6f}"
            logging.info(msg)

    summary_writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def validate(config, testloader, full_model, writer_dict):
    full_model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS # 2
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, name = batch
            size = label.size()
            if IS_MAC:
                image = image.to(device)
                label = label.long().to(device)
                bd_gts = bd_gts.float().to(device)
            else:
                image = image.cuda()
                label = label.long().cuda()
                bd_gts = bd_gts.float().cuda()

            losses, pred, _, _ = full_model(image, label, bd_gts)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                print("i: ", i)
                x = F.interpolate(input=x,
                                  size=size[-2:],
                                  mode='bilinear',
                                  align_corners=config.MODEL.ALIGN_CORNERS)

                confusion_matrix[..., i] += get_confusion_matrix(
                    label, x, size, config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL)
                if i == 1:
                    print('yaho')
                    pred2 = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
                    for i, color in enumerate(color_map):
                        for j in range(3):
                            # 9 잔디
                            #
                            # if i not in [9, 11]:
                            #     continue
                            sv_img[:, :, j][pred2 == i] = color[j]
                    sv_img = Image.fromarray(sv_img)
                    sv_img = concatenate_two_images(sv_img, image)
                    sv_path = os.path.join(config.ROOT, 'val')
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                    sv_img.save(sv_path + name)
            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            ave_loss.update(loss.item())

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()

        logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    summary_writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    summary_writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    summary_writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config,
            test_dataset,
            testloader,
            model,
            sv_dir='./',
            sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            if IS_MAC:
                image_cuda = image.to(device)
                # label = label.long().to(device)
            else:
                image_cuda = image.cuda()
                # label = label.long().cuda()
            pred = test_dataset.single_scale_inference(config, model,
                                                       image_cuda)

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(pred,
                                     size[-2:],
                                     mode='bilinear',
                                     align_corners=config.MODEL.ALIGN_CORNERS)

            confusion_matrix += get_confusion_matrix(label, pred, size,
                                                     config.DATASET.NUM_CLASSES,
                                                     config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model, sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(config, model,
                                                       image.cuda())

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(pred,
                                     size[-2:],
                                     mode='bilinear',
                                     align_corners=config.MODEL.ALIGN_CORNERS)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
