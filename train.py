import argparse
import numpy as np
import torch
import pandas as pd
import datetime
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from utils.utils import *
from utils.misc import setup_cfg, lossmeter
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
import trainers.alp as alp
from utils.utils import *
import os
from dassl.utils import Registry
from datasets import RESISC45
from datasets import Optimal31
from datasets import MLRSNet
from datasets import UCM
from datasets import AID
from datasets import PatternNet
from datasets import WHURS19

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def test(args, teloader, model):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_pl = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    one_hot_pl = []

    for i, (inputs) in enumerate(tqdm(teloader)):
        img = inputs["img"]
        labels = inputs["label"]

        if args.zero_shot:
            with torch.no_grad():
                output_pseudo_label = model(inputs.cuda(), zero_shot=True)
                _, predicted_pl = output_pseudo_label.max(1)
                one_hot_pl.append(predicted_pl.eq(labels.cuda()).cpu())
                acc1_pl = one_hot_pl[-1].sum().item() / len(labels)
                top1_pl.update(acc1_pl, len(labels))

        else:
            with torch.no_grad():
                inputs, labels = img.cuda(), labels.cuda()
                outputs = model(inputs, clip_eval=True)
                _, predicted = outputs.max(1)
                one_hot.append(predicted.eq(labels).cpu())
                acc1 = one_hot[-1].sum().item() / len(labels)
                top1.update(acc1, len(labels))

    if not args.zero_shot:
        return top1.avg * 100, top1_pl.avg * 100
    else:
        return top1_pl.avg * 100

def train_lafter(args, model, tr_loader, val_loader, test_loader=None):

    
    # # Print learnable parameters
    # print('<<<<<<<<<<<<<<<<<<<<<<Learnable Parameters>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # for name, param in model.model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
            
    # model.init_vision_adapter()

    batch_time = lossmeter()
    data_time = lossmeter()
    best_acc = 0
    columns = ['Epoch', 'PS Text Acc','PS ZS Acc', 'Epoch Loss', 'Validation Accuracy', 'Test Accuracy','Best Model']
    # df = pd.DataFrame(columns=columns)
    df_to_append = []

    # Initialize early stopping parameters
    early_stopping_counter = 0
    early_stopping_threshold = 20

    # if args.dataset=="mlrsnet" or args.dataset=="aid":
    #     early_stopping_threshold=10

    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        model.eval()
        model.adapter.train()
        end = time.time()
        

        pl_text_acc = lossmeter()
        pl_zs_acc = lossmeter()
        pl_svl_acc = lossmeter()
        pl_vision_adapter_acc = lossmeter()
        pl_scalemae_acc = lossmeter()
        pl_satmae_acc = lossmeter()
        total_loss = lossmeter()
        pl_dino_adapter_acc = lossmeter()
        pl_clip_adapter_acc = lossmeter()
        zs_conf=lossmeter()
        text_conf=lossmeter()

        for i, batch in enumerate((tr_loader)):
            data_time.update(time.time() - end)
            batch_time.update(time.time() - end)

            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(model.device)

            optimizer.zero_grad()

            with torch.no_grad():
                output_text = model.forward_normal_for_pl(input[0])

            if args.without_prompts:
                print("Without Prompts")
                out = model.    d(input[1].float().cuda())
            else:
                out = model.forward_aug_with_prompts(input[1].float().cuda())

            pseudo_label = F.softmax(output_text, dim=-1)  # / 0.04

            text_max_confs = pseudo_label.max(dim=1).values.float()
            text_average_conf = torch.mean(text_max_confs)
            text_conf.update(text_average_conf.item(), len(batch["label"]))
            
            pseudo_label = pseudo_label.argmax(dim=1, keepdim=True)
            pseudo_label = pseudo_label.flatten().cuda()
            pl_text_acc.update((pseudo_label == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))

            if not args.text_only:
                # clip_conf = output_zs.softmax(dim=-1).max(dim=-1).values.mean().item

                if args.pl_technique=="pl_text":
                    # Get Pseudo Label from Zero-Shot
                    with torch.no_grad():
                        output_zs = model.forward_pl_zeroshot(input[0])
                        pseudo_label_zero_shot = F.softmax(output_zs, dim=-1)

                        zs_max_confs = pseudo_label_zero_shot.max(dim=1).values.float()
                        zs_average_conf = torch.mean(zs_max_confs)
                        zs_conf.update(zs_average_conf.item(), len(batch["label"]))

                        pseudo_label_zero_shot = pseudo_label_zero_shot.argmax(dim=1, keepdim=True)
                        pseudo_label_zero_shot = pseudo_label_zero_shot.flatten().cuda()
                        pl_zs_acc.update((pseudo_label_zero_shot == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))

                    if args.bws=="fixed_alpha":
                        # Choose a value for alpha in the range [0, 1]
                        alpha = args.bws.split('_')[-1]
                        combined_tensor = alpha * output_zs + (1 - alpha) * output_text
                        pseudo_label_pre_softmax = torch.mean(combined_tensor, dim=1)
                        pseudo_label_comb  = F.softmax(pseudo_label_pre_softmax, dim=0)
                        pseudo_label_comb  = pseudo_label_comb .argmax()
                        if pseudo_label_comb .dim() > 0:
                            pseudo_label = pseudo_label_comb .view(-1)

                    elif args.bws=="avg":
                        # Combine the tensors along a new dimension (e.g., concatenate along a new dimension)
                        combined_tensor = torch.stack([output_zs, output_text], dim=2)
                        # Average along the new dimension
                        pseudo_label_pre_softmax = torch.mean(combined_tensor, dim=2)
                        pseudo_label_comb = F.softmax(pseudo_label_pre_softmax, dim=1)
                        pseudo_label_comb = pseudo_label_comb.argmax(dim=1)
                        # Ensure pseudo_label_text is 1D or flatten it
                        if pseudo_label_comb.dim() > 1:
                            pseudo_label = pseudo_label_comb.view(-1)

                    elif args.bws=="conf_alpha":
                        # BWS Computation: Alpha = softmax(concat(pl_zs,pl_text))
                        alpha = torch.cat([torch.max(F.softmax(output_zs),dim=1)[0].unsqueeze(1),torch.max(F.softmax(output_text),dim=1)[0].unsqueeze(1)], dim=-1)
                        alpha = F.softmax(alpha, dim=-1)
                        # New Psuedo Label
                        pseudo_label_pre_softmax = (output_zs*alpha[:, 0].unsqueeze(1) +  output_text*alpha[:, 1].unsqueeze(1))
                        pseudo_label_conf_alpha = torch.flatten(F.softmax(pseudo_label_pre_softmax, dim=-1).argmax(dim=1, keepdim=True))
                        #Change later
                        pseudo_label = pseudo_label_conf_alpha
                    else:
                        raise NotImplementedError
                    
                elif args.pl_technique=="dino_adapter":

                    with torch.no_grad():
                        output_dino_adapter = model.forward_dino_adapter(input[0])
                        pseudo_label_pre_softmax = output_dino_adapter
                        pseudo_label_dino_adapter = F.softmax(output_dino_adapter, dim=-1).argmax(dim=1, keepdim=True)
                        pseudo_label_dino_adapter = pseudo_label_dino_adapter.flatten().cuda()
                        pl_dino_adapter_acc.update((pseudo_label_dino_adapter == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))
                        pseudo_label = pseudo_label_dino_adapter

                    
                    
                elif args.pl_technique=="pl_svl":
                    # print (" Get Pseudo Label from SVL")
                    # Get Pseudo Label from SVL
                    with torch.no_grad():
                        output_svl = model.forward_svl(input[0])
                        pseudo_label_svl = F.softmax(output_svl, dim=-1).argmax(dim=1, keepdim=True)
                        pseudo_label_svl = pseudo_label_svl.flatten().cuda()
                        pl_svl_acc.update((pseudo_label_svl == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))

                    if args.bws=="avg":
                        # Combine the tensors along a new dimension (e.g., concatenate along a new dimension)
                        combined_tensor = torch.stack([output_text, output_svl], dim=2)
                        # Average along the new dimension
                        pseudo_label_pre_softmax = torch.mean(combined_tensor, dim=2)
                        pseudo_label_text = F.softmax(pseudo_label_pre_softmax, dim=1)
                        pseudo_label_text = pseudo_label_text.argmax(dim=1)
                        # Ensure pseudo_label_text is 1D or flatten it
                        if pseudo_label_text.dim() > 1:
                            pseudo_label = pseudo_label_text.view(-1)

                    elif args.bws=="conf_alpha":
                        # BWS Computation: Alpha = softmax(concat(pl_zs,pl_text))
                        alpha = torch.cat([torch.max(F.softmax(output_text),dim=1)[0].unsqueeze(1),torch.max(F.softmax(output_svl),dim=1)[0].unsqueeze(1)], dim=-1)
                        alpha = F.softmax(alpha, dim=-1)
                        # New Psuedo Label
                        pseudo_label_pre_softmax = (output_text*alpha[:, 0].unsqueeze(1) +  output_svl*alpha[:, 1].unsqueeze(1))
                        pseudo_label_conf_alpha = torch.flatten(F.softmax(pseudo_label_pre_softmax, dim=-1).argmax(dim=1, keepdim=True))
                        #Change later
                        pseudo_label = pseudo_label_conf_alpha
                        
                    else:
                        raise NotImplementedError
                
                elif args.pl_technique=="pl_text_svl":
                    with torch.no_grad():
                        output_zs = model.forward_pl_zeroshot(input[0])
                        pseudo_label_zero_shot = F.softmax(output_zs, dim=-1).argmax(dim=1, keepdim=True)
                        pseudo_label_zero_shot = pseudo_label_zero_shot.flatten().cuda()
                        pl_zs_acc.update((pseudo_label_zero_shot == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))
                    
                    with torch.no_grad():
                        output_svl = model.forward_svl(input[0])
                        pseudo_label_svl = F.softmax(output_svl, dim=-1).argmax(dim=1, keepdim=True)
                        pseudo_label_svl = pseudo_label_svl.flatten().cuda()
                        pl_svl_acc.update((pseudo_label_svl == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))

                    if args.bws=="avg":
                        # Combine the tensors along a new dimension (e.g., concatenate along a new dimension)
                        combined_tensor = torch.stack([output_zs, output_text, output_svl], dim=2)
                        # Average along the new dimension
                        pseudo_label_pre_softmax = torch.mean(combined_tensor, dim=2)
                        pseudo_label_text = F.softmax(pseudo_label_pre_softmax, dim=1)
                        pseudo_label_text = pseudo_label_text.argmax(dim=1)
                        # Ensure pseudo_label_text is 1D or flatten it
                        if pseudo_label_text.dim() > 1:
                            pseudo_label = pseudo_label_text.view(-1)

                    elif args.bws=="conf_alpha":
                        # BWS Computation: Alpha = softmax(concat(pl_zs,pl_text))
                        alpha = torch.cat([torch.max(F.softmax(output_zs),dim=1)[0].unsqueeze(1),torch.max(F.softmax(output_text),dim=1)[0].unsqueeze(1),torch.max(F.softmax(output_svl),dim=1)[0].unsqueeze(1)], dim=-1)
                        alpha = F.softmax(alpha, dim=-1)
                        # New Psuedo Label
                        pseudo_label_pre_softmax = (output_zs*alpha[:, 0].unsqueeze(1) +  output_text*alpha[:, 1].unsqueeze(1) + output_svl*alpha[:, 2].unsqueeze(1))
                        pl_new = torch.flatten(F.softmax(pseudo_label_pre_softmax, dim=-1).argmax(dim=1, keepdim=True))
                        #Change later
                        pseudo_label = pl_new 

                    else:
                        raise NotImplementedError

                elif args.pl_technique=="svl_only":
                    with torch.no_grad():
                        output_svl = model.forward_svl(input[0])
                        pseudo_label_pre_softmax = output_svl
                        pseudo_label_svl = F.softmax(output_svl, dim=-1).argmax(dim=1, keepdim=True)
                        pseudo_label_svl = pseudo_label_svl.flatten().cuda()
                        pl_svl_acc.update((pseudo_label_svl == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))
                        pseudo_label = pseudo_label_svl

                elif args.pl_technique=="vision_adapter":
                    with torch.no_grad():
                        output_vision_adapter = model.forward_vision_adapter(input[0])
                        pseudo_label_pre_softmax = output_vision_adapter
                        pseudo_label_vision_adapter = F.softmax(output_vision_adapter, dim=-1).argmax(dim=1, keepdim=True)
                        pseudo_label_vision_adapter = pseudo_label_vision_adapter.flatten().cuda()
                        pl_vision_adapter_acc.update((pseudo_label_vision_adapter == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))
                        pseudo_label = pseudo_label_vision_adapter

                    if args.bws=="avg":
                        # Combine the tensors along a new dimension (e.g., concatenate along a new dimension)
                        combined_tensor = torch.stack([output_text, output_vision_adapter], dim=2)
                        # Average along the new dimension
                        pseudo_label_pre_softmax = torch.mean(combined_tensor, dim=2)
                        pseudo_label_text = F.softmax(pseudo_label_pre_softmax, dim=1)
                        pseudo_label_text = pseudo_label_text.argmax(dim=1)
                        # Ensure pseudo_label_text is 1D or flatten it
                        if pseudo_label_text.dim() > 1:
                            pseudo_label = pseudo_label_text.view(-1)

                elif args.pl_technique=="scalemae":
                    with torch.no_grad():
                        output_scalemae = model.forward_scalemae(input[0])
                        pseudo_label_pre_softmax = output_scalemae
                        pseudo_label_scalemae = F.softmax(output_scalemae, dim=-1).argmax(dim=1, keepdim=True)
                        pseudo_label_scalemae = pseudo_label_scalemae.flatten().cuda()
                        pl_scalemae_acc.update((pseudo_label_scalemae == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))
                        pseudo_label = pseudo_label_scalemae

                elif args.pl_technique=="satmae":
                    with torch.no_grad():
                        output_satmae = model.forward_satmae(input[0])
                        pseudo_label_pre_softmax = output_satmae
                        pseudo_label_satmae = F.softmax(output_satmae, dim=-1).argmax(dim=1, keepdim=True)
                        pseudo_label_satmae = pseudo_label_satmae.flatten().cuda()
                        pl_satmae_acc.update((pseudo_label_satmae == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))
                        pseudo_label = pseudo_label_satmae

                elif args.pl_technique=="clip_adapter":
                    with torch.no_grad():
                        output_clip_adapter = model.forward_clip_adapter(input[0])
                        pseudo_label_pre_softmax = output_clip_adapter
                        pseudo_label_clip_adapter = F.softmax(output_clip_adapter, dim=-1).argmax(dim=1, keepdim=True)
                        pseudo_label_clip_adapter = pseudo_label_clip_adapter.flatten().cuda()
                        pl_clip_adapter_acc.update((pseudo_label_clip_adapter == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))
                        pseudo_label = pseudo_label_clip_adapter
                    
                
                else:
                    raise NotImplementedError
        
            if args.loss_fn=="crossentropy":
                loss = criteria(out.squeeze(), pseudo_label)
            elif args.loss_fn=="distill":
                loss = criteria(out.squeeze(), pseudo_label_pre_softmax)    
            total_loss.update(loss.item(),len(tr_loader))
            # total_loss.update(loss.item(),len(tr_loader))

            if i % args.print_freq == 0:
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "loss {losses}\t"
                    "lr {lr:.6e}".format(
                        epoch,
                        args.epochs,
                        i + 1,
                        len(tr_loader),
                        losses=loss.item(),
                        lr=optimizer.param_groups[0]["lr"],
                    ))

            loss.backward()
            optimizer.step()
        scheduler.step()
        
        print(f'Text classifier average confiderence: {text_conf.avg}')
        print(f'Zero shot average confiderence: {zs_conf.avg}')
        
        print(f'Epoch Loss: {total_loss.avg}')

        print(f'Evaluation: {epoch}')
        val_acc = test_prompting(val_loader, model)
        print(f'TOP-1 Accuracy: {val_acc}')
        all_acc.append(val_acc)

        # ps_text_acc=pl_text_acc.avg
        # ps_zs_acc=pl_zs_acc.avg

        print(f'Pseudo Label Text Accuracy: {pl_text_acc.avg}')
        print(f'Pseudo Label Zero Shot Accuracy: {pl_zs_acc.avg}')
        print(f'Pseudo Label SVL Accuracy: {pl_svl_acc.avg}')
        # print(f'Pseudo Label Vision Adapter Accuracy: {pl_vision_adapter_acc.avg}')
        # print(f'Pseudo Label ScaleMAE Accuracy: {pl_scalemae_acc.avg}')
        # print(f'Pseudo Label SATMAE Accuracy: {pl_satmae_acc.avg}')
        print(f'Pseudo Label DINO Adapter Accuracy: {pl_dino_adapter_acc.avg}')
        print(f'Pseudo Label CLIP Adapter Accuracy: {pl_clip_adapter_acc.avg}')

        # if args.wandb_run_name:
        #     wandb.log({"Epoch": epoch, "PS Text Acc": pl_text_acc.avg, "PS ZS Acc": pl_zs_acc.avg, "Epoch Loss": total_loss.avg, "Validation Accuracy": val_acc})

        best_test_acc=None
        if val_acc>best_acc:
            best_val_acc="Yes"
            best_acc=val_acc
            early_stopping_counter = 0
            print('------------')
            print("Best Epoch ", epoch)
            print("Best Val acc", val_acc)
            best_test_acc = test_prompting(test_loader, model)
            print("Test acc ", best_test_acc)
            print('------------')

            #Save the whole model
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pth")) 
        else:
            early_stopping_counter += 1

        # df_to_append.append([epoch, ps_text_acc, ps_zs_acc, total_loss.avg, val_acc, best_test_acc, best_val_acc])
        print("Output directory ",args.output_dir)

        if early_stopping_counter >= early_stopping_threshold:
            print(f'Early stopping at epoch {epoch} due to no improvement in validation accuracy.')
            break
    df = pd.DataFrame(df_to_append, columns=columns)    
    csv_path = os.path.join(args.output_dir, "training_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f'-------------------------------- Best Validation Accuracy: {max(all_acc)} --------------------------------')
    print(f'-------------------------------- Best Validation Accuracy Epoch: {all_acc.index(max(all_acc))} --------------------------------')
    
def main(args):

    setup_seed(cfg.SEED)
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed

    dataset_name = args.dataset

    # setup_txt_epochs(args, dataset_name)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    
    dataset_registary = Registry("Dataset")

    dataset_registary.register(RESISC45)
    dataset_registary.register(UCM)
    dataset_registary.register(WHURS19)
    dataset_registary.register(MLRSNet)
    dataset_registary.register(Optimal31)

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
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_project_name', type=str, default=None)
    parser.add_argument('--finetuned_dino', action="store_true")
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

