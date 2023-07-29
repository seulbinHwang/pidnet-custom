# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pprint

import logging
import timeit
import random

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BoundaryLoss
from utils import function
from utils.utils import create_logger, FullModel
import platform
from typing import Tuple

import torch


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


def load_pretrained(model, pretrained_directory):
    pretrained_dict = torch.load(pretrained_directory, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {
        k[6:]: v
        for k, v in pretrained_dict.items()
        if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)
    }
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    return model


# python tools/train.py --cfg configs/ade/pidnet_large_ade_fine_tune.yaml --enable_fine_tune True GPUS "(0,)" TRAIN.BATCH_SIZE_PER_GPU 4
# python tools/train.py --cfg configs/ade/pidnet_large_ade.yaml GPUS "(0,)" TRAIN.BATCH_SIZE_PER_GPU 4
def parse_args():
    # python tools/train.py --cfg configs/cityscapes/pidnet_large_ade.yaml GPUS "(0,1)" TRAIN.BATCH_SIZE_PER_GPU 6
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument(
        '--cfg',
        help='experiment configure file name',
        default=
        "configs/ade/pidnet_large_ade.yaml",  #"configs/cityscapes/pidnet_large_cityscapes.yaml",# #, #  #
        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('--enable_fine_tune', type=bool, default=False)
    parser.add_argument('--pretrained_model_directory',
                        help='dir for pretrained model',
                        default='./pretrained_models/cityscapes/best_3_rider.pt',
                        type=str)
    parser.add_argument('--low_resolution', type=bool, default=False)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def select_gpus(config) -> Tuple[int]:
    if IS_MAC:
        gpus = (0, )
    else:
        gpus = list(config.GPUS)
        device_num = torch.cuda.device_count()
        if device_num == 1:
            gpus = (0, )
        elif device_num != len(gpus):
            print(f"The gpu numbers do not match!, chosen gpus: {gpus}")
    return gpus


def make_model_and_load_param(args) -> nn.Module:
    pretrain_exists = 'imagenet' in config.MODEL.PRETRAINED
    model = models.pidnet.get_seg_model(config,
                                        imgnet_pretrained=pretrain_exists)

    ###################
    if args.enable_fine_tune:
        model = load_pretrained(model, args.pretrained_model_directory)
        # Freeze all layers except the last head
        requires_grad_name = ["seghead_p", "seghead_d", "final_layer"]
        for name, param in model.named_parameters():
            param.requires_grad = False
            for requires_grad_name_ in requires_grad_name:
                if requires_grad_name_ in name:
                    param.requires_grad = True
                    break
    return model


def main():
    args = parse_args()
    if args.seed > 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    """
    final_output_dir: 
        "output/ade/pidnet_large_ade_4_time"
    tb_log_dir:
        "log/ade/pidnet_large/pidnet_large_ade_2023-07-19-23-01"
    """
    logger, final_output_dir, tb_log_dir = create_logger(cfg=config,
                                                         cfg_name=args.cfg,
                                                         phase='train')
    logger.info(pprint.pformat(args))
    logger.info(config)
    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting.
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    gpus = select_gpus(config)
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    model = make_model_and_load_param(args)

    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # 1024, 1024
    train_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,  # data/
        list_path=config.DATASET.TRAIN_SET,  # list/cityscapes/train.lst
        num_classes=config.DATASET.NUM_CLASSES,  # 2, 3, 4
        multi_scale=config.TRAIN.MULTI_SCALE,  # True
        flip=config.TRAIN.FLIP,  # True
        ignore_label=config.TRAIN.IGNORE_LABEL,  # 255
        base_size=config.TRAIN.BASE_SIZE,  # 2048
        crop_size=crop_size,  # (1024, 1024)
        scale_factor=config.TRAIN.SCALE_FACTOR,
        low_resolution=args.low_resolution)  # 16
    """
        train_dataset: 
            학습에 사용할 데이터셋 객체입니다. 
            이 데이터셋은 trainloader에 전달되어 배치로 분할됩니다.
        batch_size: 
            배치의 크기를 결정하는 매개변수입니다. 
            학습 중에 사용될 각 배치의 샘플 수를 지정합니다.
        shuffle: 
            데이터를 섞을지 여부를 결정하는 매개변수입니다. 
            True로 설정하면 매 에폭마다 데이터가 섞여 다양한 순서로 모델에 공급됩니다.
        num_workers: 
            데이터를 로드하기 위해 사용되는 워커(스레드)의 수입니다. 
            병렬적으로 데이터를 로드하여 학습 속도를 향상시킬 수 있습니다.
        pin_memory: 
            CUDA 호환 GPU에서 데이터를 로드할 때 CPU 메모리를 고정할지 여부를 결정하는 매개변수입니다. 
            일반적으로 True로 설정하여 raw_data 로드 속도를 높일 수 있습니다.
        drop_last: 
            마지막 배치가 배치 크기보다 작을 경우 해당 배치를 버릴지 여부를 결정하는 매개변수입니다. 
            True로 설정하면 마지막 배치가 작을 때 해당 배치를 버립니다.
    """
    # batch_size: 6 * 1
    if IS_MAC:
        num_workers = 0
        pin_memory = True
    else:
        num_workers = config.WORKERS
        pin_memory = False
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,  # 6* 1
        shuffle=config.TRAIN.SHUFFLE,  # True
        num_workers=num_workers,  # 6
        pin_memory=pin_memory,
        drop_last=True)
    # 1024, 2048
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,  # data/
        list_path=config.DATASET.TEST_SET,  # list/cityscapes/val.lst
        num_classes=config.DATASET.NUM_CLASSES,  # 2
        multi_scale=config.TEST.MULTI_SCALE,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,  # 255
        base_size=config.TEST.BASE_SIZE,  # 2048
        crop_size=test_size,
        low_resolution=args.low_resolution)  # (1024, 2048)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),  # 6
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False)

    # criterion
    # True
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(
            ignore_label=config.TRAIN.
            IGNORE_LABEL,  # config.TRAIN.IGNORE_LABEL,  # 255
            thres=config.LOSS.OHEMTHRES,  # 0.9
            min_kept=config.LOSS.OHEMKEEP,  # 131072
            weight=train_dataset.class_weights)  # [ 1.0023,0.9843, ]
    else:
        sem_criterion = CrossEntropy(
            ignore_label=config.TRAIN.
            IGNORE_LABEL,  # config.TRAIN.IGNORE_LABEL,  # 255
            weight=train_dataset.class_weights)  # [ 1.0023,0.9843, ]

    bd_criterion = BoundaryLoss()

    full_model = FullModel(model, sem_loss=sem_criterion, bd_loss=bd_criterion)
    if IS_MAC:
        full_model = nn.DataParallel(full_model).to(device)
    else:
        full_model = nn.DataParallel(full_model, device_ids=gpus).cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(full_model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(
            params,
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV,
        )
    else:
        raise ValueError('Only Support SGD optimizer')
    # 1200개 / 6 = 200 (epoch_iters)
    epoch_iters = int(train_dataset.__len__() /
                      config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    best_mIoU = 0
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME
    # False
    if config.TRAIN.RESUME:
        # TODO: final_output_dir 을 "pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt 로?"
        # "output/cityscapes/pidnet_large_cityscapes/checkpoint.pth.tar"
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            # map_location -> 'cuda:0' 장치에 저장된 모델을 CPU로 로드하고 사용합니다.
            checkpoint = torch.load(model_state_file,
                                    map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']

            full_model.module.model.load_state_dict({
                k.replace('model.', ''): v
                for k, v in dct.items() if k.startswith('model.')
            })
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(
                checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH  # 484
    num_iters = config.TRAIN.END_EPOCH * epoch_iters  # 484 * 200 = 96800
    real_end = 120 + 1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    eval_save_dir = os.path.join(final_output_dir, 'eval')
    # real_end: end_epoch = 484
    for epoch in range(last_epoch, real_end):
        """
        여기서 sampler는 DataLoader가 데이터셋에서 어떤 샘플을 선택할지를 결정하는 역할을 합니다. 
        따라서 이 코드는 sampler가 에폭(epoch)에 따라 데이터를 다르게 선택하도록 설정하는 부분입니다.
        """
        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(
                current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)
        num_epoch = config.TRAIN.END_EPOCH  # 484
        base_lr = config.TRAIN.LR  # 0.01
        function.train(
            config,
            epoch,
            num_epoch,  # 484
            epoch_iters,  # 1200 / batch_size (6) = 200
            base_lr,
            num_iters,  # num_epoch(484) * epoch_iters(200) = 96800
            trainloader,  # torch.utils.data.DataLoader
            optimizer,  # torch.optim.SGD
            full_model,
            writer_dict)
        if flag_rm == 1 or (epoch % 5 == 0 and epoch < real_end - 100) or (
                epoch >= real_end - 100):
            valid_loss, mean_IoU, IoU_array = function.validate(
                config, testloader, full_model, writer_dict, eval_save_dir)
        if flag_rm == 1:
            flag_rm = 0

        logger.info('=> saving checkpoint to {}'.format(final_output_dir +
                                                        'checkpoint.pth.tar'))
        torch.save(
            {
                'epoch': epoch + 1,
                'best_mIoU': best_mIoU,
                'state_dict': full_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))
        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(full_model.module.state_dict(),
                       os.path.join(final_output_dir, 'best.pt'))
        msg = 'Loss: {:.3f}, Mean_IoU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
            valid_loss, mean_IoU, best_mIoU)
        logging.info(msg)
        logging.info(IoU_array)

    torch.save(full_model.module.state_dict(),
               os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % np.int((end - start) / 3600))
    logger.info('Done')


# python tools/train.py --cfg configs/cityscapes/pidnet_large_ade.yaml TRAIN.BATCH_SIZE_PER_GPU 6
if __name__ == '__main__':
    main()
