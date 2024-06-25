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

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):

    setup_seed(cfg.SEED)
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed

    dataset_name = args.dataset

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    
    trainer = build_trainer(cfg)
    trainer.train(cfg)
    trainer.test(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument('--exp-name', type=str, required=False)
    parser.add_argument('--scheduler', default='cosine')
    parser.add_argument('--scheduler-epochs', type=int, default=15)
    parser.add_argument('--scheduler-gamma', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--acc-batches', type=int, default=1)
    parser.add_argument('--arch', type=str, default='ViT-B/32', required=False)
    parser.add_argument('--gpt_prompts', action='store_true')
    parser.add_argument('--text_prompts', action='store_true')
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--txt_cls', type=str, default='lafter', required=False, choices=['cls_only',
                                                                                      'templates_only', 'lafter', 'zero_shot'])
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--txt_epochs', type=int, default=1000)
    parser.add_argument('--logfolder', default='logs', type=str)
    parser.add_argument('--text_only', action="store_true")
    parser.add_argument('--bws', type=str, default='None', choices=['conf_alpha','fixed_alpha_0.25', 'avg', 'None'])
    parser.add_argument('--ln_frozen', action="store_true")
    parser.add_argument('--loss_fn', default='crossentropy')
    parser.add_argument('--train_text_ln', action="store_true")
    parser.add_argument('--desc_noise', type=float, default=0.0)
    parser.add_argument('--classifer_random_weights', action="store_true")
    parser.add_argument('--ve_unshared', action="store_true")
    parser.add_argument('--desc_emb', action="store_true")
    # parser.add_argument('--svl_pl', action="store_true")
    parser.add_argument('--svl_model_path', type=str, default=None)
    parser.add_argument('--pl_technique', type=str, default='None', choices=['clip_adapter','dino_adapter','None','pl_text', 'pl_svl', 'pl_text_svl','svl_only','vision_adapter','scalemae', 'satmae'])
    parser.add_argument('--dataset',type=str, required=False, default='resisc45')
    parser.add_argument('--configuration',type=str, required=False, default='vit_b32', choices=['GeoRSCLIP_b_32','GeoRSCLIP_l_14', 'vit_b32','GeoRSCLIP_adapter'])
    parser.add_argument('--vision_adapter', action="store_true")
    parser.add_argument('--scalemae_path', type=str, default=None, required=False)
    parser.add_argument('--satmae_path', type=str, default=None)
    parser.add_argument('--ssl_enc_path', type=str, default=None)
    parser.add_argument('--diff_encoder', type=str, default=None)
    parser.add_argument('--without_prompts', action="store_true")
    parser.add_argument('--classifier_frozen_weights', action="store_true")
    parser.add_argument('--test_only', action="store_true")
    parser.add_argument('--k_desc', type=int, default=0)
    parser.add_argument('--save_emb', action="store_true")

    args = parser.parse_args()

    if args.dataset=="mlrsnet" or args.dataset=="imagenet":
        print("Setting batch size to 512 and lr to 0.004")
        args.batch_size=512
        args.lr=0.004

    cfg = setup_cfg(args)

    main(args)

