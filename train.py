import argparse
import numpy as np
import torch
import pandas as pd
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from utils.utils import *
from utils.misc import setup_cfg
# custom
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.aid
import datasets.food101
import datasets.sun397
import datasets.ucf101
import datasets.imagenet_r
import datasets.imagenet
import datasets.imagenet_s
import datasets.imagenet_a
import datasets.caltech101
import datasets.cifar
import datasets.resisc45
import datasets.aid
import datasets.optimal31
import datasets.mlrsnet
import datasets.ucm

import trainers.ALP as ALP

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # Use the first GPU


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def main(cfg):

    setup_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    
    trainer = build_trainer(cfg)
    trainer.train(max_epoch=cfg.OPTIM.MAX_EPOCH, cfg=cfg)
    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/text_cls/vit_b32.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/imagenet.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument('--txt_cls', type=str, default='lafter', required=False, choices=['cls_only',
                                                                                      'templates_only', 'lafter', 'zero_shot'])

    parser.add_argument('--config', type=str, default='config.yml', required=False, help='config file')
    args = parser.parse_args()

    cfg = setup_cfg(args)
    cfg.merge_from_file(args.config)

    main(cfg)

