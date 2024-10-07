from torch.nn import Dropout
import math
import os.path as osp
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import read_image
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
from clip.clip import tokenize
from torch.utils.data import Dataset, DataLoader
import time
import sys
sys.path.append("../")
from utils.model_utils import *
from utils.utils import *
from functools import reduce
from operator import mul
from utils.data_utils import ds_specific_templates
import wandb

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root='all_weights')
    print(model_path)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TopKDataset(Dataset):
    def __init__(self, data, transform, top_k_idxs, top_k_conf, top_k_pseudo):
        self.data_source = data
        self.transform = transform
        self.top_k_idxs = top_k_idxs
        self.top_k_conf = top_k_conf
        self.top_k_pseudo = top_k_pseudo

    def __len__(self):
        return len(self.top_k_idxs)

    def __getitem__(self, idx):

        new_idx = self.top_k_idxs[idx]
        item = self.data_source[new_idx]

        output = {
            'label': self.top_k_pseudo[idx],
            'label_conf': self.top_k_conf[idx],
            'ground_truth': item.label,
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            img = self.transform(img0)
            output["img"] = img
        else:
            output["img"] = img0
        
        return output
    

class DictDataset(Dataset):
    def __init__(self, data_dict, transform=None):

        self.data_dict = data_dict

    def __len__(self):
        return next(iter(self.data_dict.values())).size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {key: value[idx] for key, value in self.data_dict.items()}
        return sample
    
class Taal(nn.Module):
    def __init__(self, num_classes, templates, device='cuda'):
        super(Taal, self).__init__()

        self.taal_enc =  torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.taal_enc.head = nn.Identity()
        self.taal_enc.eval()
        in_features = self.taal_enc.cls_token.shape[-1]
        self.taal_h_module = AdapterMLP(num_classes=num_classes, input_size=in_features, hidden_size=256)

    def forward(self, x):
        with torch.no_grad():
            op = self.taal_enc(x)
        op = self.taal_h_module(op)
        return op
    
class AdapterMLP(nn.Module):
    '''
    MLP Network for low-shot adaptation (trained on top of frozen features)
    '''
    def __init__(self,num_classes,input_size,hidden_size):
        super(AdapterMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        out = self.mlp(x)
        return out
      
class NoLAUFT(nn.Module):

    def __init__(self, clip_model, classes, templates, ds_templates=None, cfg=None):
        super(NoLAUFT, self).__init__()

        self.device = cfg.DEVICE
        self.cfg = cfg
        self.dataset_templates = ds_templates
        self.classes = classes
        self.dataset_name = cfg.DATASET.NAME
        self.txt_cls = cfg.txt_cls
        self.templates = templates

        self.backbone_out_size = 512 # todo for ViT-L use 768 - for ViT-B use 512
        self.hidden_size = 768 # todo for ViT-L use 1024 - for ViT-B use 768
        self.num_tokens = cfg.NUM_TOKENS # todo all experiments are run with num_tokens = 50
        
        self.prompt_proj = nn.Identity()
        prompt_dim = self.hidden_size
        
        self.prompt_dropout = Dropout(0.0)
        self.clip_model = clip_model.to(cfg.DEVICE)
        self.taal = Taal(num_classes=len(classes), templates=templates, device=cfg.DEVICE)

        self.cde_adapter = nn.Sequential(nn.Linear(int(self.backbone_out_size), len(classes), bias=False))

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, self.hidden_size), requires_grad=True)
        patch_size = (16, 16)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
      
        self.avg_text_emb = self.gen_emb()  # Average text embeddings for each class
        self.cde_adapter[0].weight.data = self.avg_text_emb.T

        
    def gen_emb(self):
        if os.path.isfile(f'embeddings/{self.txt_cls}_{self.dataset_name}.pt'):
            print('******** Loading Already Saved Averaged Text Embeddings *********')
            return torch.load(f'embeddings/{self.txt_cls}_{self.dataset_name}.pt')
        
        path_to_file = f'./descriptions/generic/{self.dataset_name}.json'
        
        print(path_to_file)
        with open(path_to_file) as f:
            gpt3_prompts = json.load(f)
        desc, labels_for_descriptions = gen_labels_with_descrptions(self.classes, descriptions=gpt3_prompts)
        templates, labels_for_templates = gen_labels_with_templates(self.classes, descriptions=self.dataset_templates)
        desc += templates
        labels_for_descriptions += labels_for_templates

        Path(f'embeddings').mkdir(parents=True, exist_ok=True)

        zeroshot_weights = []
        with torch.no_grad():
            for c in range(len(self.classes)):
                text = [desc[idx.item()] for idx in (torch.tensor(labels_for_descriptions).cuda()==c).nonzero()]
                # text = [template.format(classname) for template in self.templates]  # format with class
                text = tokenize(text).cuda()  # tokenize # (50, 77) --> 50 templates/texts from GPT
                class_embeddings = self.clip_model.encode_text(text)  # embed with text encoder # (50, 512) --> embeddings for all 50 texts
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # L2 norm of the embeddings (dim 2)
                class_embedding = class_embeddings.mean(dim=0, keepdim=True)
                class_embedding /= class_embedding.norm()

                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights).cuda()  # (512, 10) --> 512 embeddings for 10 classes'
            zeroshot_weights =  zeroshot_weights.squeeze(1)
            zeroshot_weights = zeroshot_weights.T
            
            torch.save((zeroshot_weights), f'embeddings/{self.txt_cls}_{self.dataset_name}.pt')
        return zeroshot_weights

    def image_features(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def eval_clip(self, x):
        with torch.no_grad():
            img_features_2 = self.incorporate_prompt(x)
            img_features_2 = self.embeddings_after_prompts(img_features_2)
            zs_emb = self.cde_adapter(img_features_2)
        return zs_emb

    # def forward(self, x):
    #     # only used for 0-shot-eval
    #     with torch.no_grad():
    #         img_features = self.image_features(x)
    #         pseudo_label = img_features.to(self.avg_text_emb.device) @ self.avg_text_emb
    #     return pseudo_label

    def forward_aug_with_prompts(self, x2):
        '''
        :param x2: the transformed image (for student)
        :return: features adapter (cls head), pseudo-labels
        '''
        img_features_2 = self.incorporate_prompt(x2)
        img_features_2 = self.embeddings_after_prompts(img_features_2)
        img_features_adapter = self.cde_adapter(img_features_2)
        return img_features_adapter

    def forward_with_text_desc(self, image, text_desc_emb):
        image_features = self.image_features(image)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_desc_emb
        logits_per_text = logit_scale * text_desc_emb.t() @ image_features.t()

        return logits_per_image, logits_per_text
    
    def forward_taal(self,x):
        return self.taal(x)


    def incorporate_prompt(self, x, teacher=False):
        B = x.shape[0]
        x = self.patch_embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        # Learnable prompts added between class token and image patches (patch embeddings?)
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        return x

    def patch_embeddings(self, x: torch.tensor):
        return self.clip_model.visual.embeddings_patch(x)

    def embeddings_after_prompts(self, x: torch.tensor):
        return self.clip_model.visual.forward_after_patch_embeddings(x)

    def positional_embeddings_for_text(self, x: torch.tensor):
        return self.clip_model.positional_embeddings(x)

    def embeddings_after_prompts_for_text(self, x: torch.tensor):
        return self.clip_model.embeddings_after_prompting(x)


@TRAINER_REGISTRY.register()
class NoLA(TrainerX):

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            wandb.init(project='NOLA', config=self.cfg, dir=log_dir, name=self.cfg.DATASET.NAME)
            self._writer = wandb

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.log({tag: scalar_value})
    
    def close_writer(self):
        if self._writer is not None:
            self._writer.finish()   

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def setup_optim(self):
        params = []
        # for key, value in self.model.named_parameters():
        #     if value.requires_grad:
        #         print("\t{}, {}, {}".format(key, value.numel(), value.shape))
        for k, v in self.model.named_parameters():
            v.requires_grad = False
            if 'prompt_embeddings' in k:
                v.requires_grad = True
                params.append((k,v))
            
            if 'visual' in k:
                if 'ln' in k or 'bn' in k:
                    v.requires_grad = True
                    params.append((k,v))

            if 'cde_adapter' in k:
                v.requires_grad = True
                params.append((k,v))
            
            if 'taal_h_module' in k:
                v.requires_grad = True
        # for key, value in self.model.named_parameters():
        #     if value.requires_grad:
        #         print("\t{}, {}, {}".format(key, value.numel(), value.shape))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

              
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params
                        if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
            {'params': [p for n, p in params
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.cfg.OPTIM.LR, betas=(0.9, 0.999))
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, None, 0.60)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, None, 0.20)
        criteria = LabelSmoothingCrossEntropy()

        return optimizer, scheduler, criteria
    
    def build_model(self):
        cfg = self.cfg
        self.patience = 15
        classnames = self.dm.dataset.classnames
        dset = cfg.DATASET.NAME
        print("Dataset: ", dset)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg.MODEL.BACKBONE.NAME)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        print("Building ZERO-SHOT-MODEL CLIP")

        text_pr = 'a photo of a {}'
        print(f"Using prompt: {text_pr}")
        self.model = NoLAUFT(clip_model=clip_model, classes=classnames,
                                          templates=[text_pr], ds_templates = ds_specific_templates[cfg.DATASET.NAME], cfg=cfg)
        
        self.model = self.model.to(self.device)
        self.optim, self.scheduler, self.criterion = self.setup_optim()
        self.register_model("adapt", self.model, self.optim, self.scheduler)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg, custom_tfm_test=te_transform, custom_tfm_train=tr_transforms)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

        return self.train_loader_x, self.val_loader, self.test_loader

    def parse_batch_train(self, batch):
        
        if isinstance(batch['img'], list):
            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(self.device, non_blocking=True)
        else:
            input = batch['img']
            input = input.to(self.device, non_blocking=True)

        label = batch["label"]
        label = label.to(self.device, non_blocking=True)
        return input, label

    def parse_batch_test(self, batch):
        input = batch["img"]
        input = input.to(self.device)
        label = batch["label"]
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {}) with acc {}'.format(name, model_path, epoch, checkpoint['val_result']))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=True)
            # self._models[name].to(self.device)
    
    def before_train(self):

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        self.init_writer(writer_dir)
        
        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

        nos_train = len(self.train_loader_x.dataset)
        nos_classes = len(self.dm.dataset.classnames)
        k=find_k(nos_train, nos_classes)

        # get top_k dataset
        top_k_dataset = self.select_top_k(k=k)

        # train taal using top_k dataset
        train_loader = DataLoader(top_k_dataset, batch_size=32, shuffle=True, num_workers=4)

        self.train_taal(train_loader)
        self.test_taal()    

        # freeze taal network
        for k, v in self.model.named_parameters():

            if 'prompt_embeddings' in k:
                v.requires_grad = True

            if 'taal' in k:
                v.requires_grad = False
            
            if 'cde_adapter' in k:
                v.requires_grad = True
            
            if 'visual' in k and ('ln' in k or 'bn' in k):
                v.requires_grad = True

    def forward_backward(self, batch_x):

        input_x, label_x = self.parse_batch_train(batch_x)
        self.model.eval()
        self.model.cde_adapter.train()
        with torch.no_grad():
            pl = self.model.taal(input_x[0].float())
            pseudo_label = F.softmax(pl, dim=-1)  # / 0.04
            pseudo_label = pseudo_label.argmax(dim=1, keepdim=True).flatten()

        output_x = self.model.forward_aug_with_prompts(input_x[1].float())
        loss_x = self.criterion(output_x.squeeze(), pseudo_label)

        self.model_backward_and_update(loss_x, names="adapt")
        self.update_lr(names="adapt")

        loss_summary = {
            "loss_x": loss_x.item(),
        }
        return loss_summary

    def select_top_k(self, k):
        # if file exists, load the embeddings
        if os.path.isfile(f'embeddings/{self.cfg.DATASET.NAME}_total_emb.pt'):
            print('******** Loading Already Saved Averaged Text Embeddings *********')
            total_emb = torch.load(f'embeddings/{self.cfg.DATASET.NAME}_total_emb.pt')
            return DictDataset(total_emb)
        
        print("******** Generating Embeddings ********")
        text_emb =  self.model.avg_text_emb

            
        # setup for top_k_selection
        total_emb = []
        idxs = []
        labels = []
        impaths = []
        train_loader = DataLoader(self.train_loader_x.dataset, batch_size=128, shuffle=False, num_workers=4)
        for batch in tqdm(train_loader):
            input = torch.stack(batch['img']).to(self.device)
            with torch.no_grad():
                temp, _ =self.model.forward_with_text_desc(input[0], text_emb)
                temp = temp.softmax(dim=-1)
            total_emb.append(temp)
            idxs.append(batch['index'])
            impaths.append(batch['impath'])
            labels.append(batch['label'])

        emb_dict = {
                'total_emb': torch.cat(total_emb),
                'idxs': torch.cat(idxs),
                'labels':  torch.cat(labels),
            }
        
        
        top_k_idxs, top_k_conf, top_k_pseudo = top_k_indices_per_class2(emb_dict, k)
        self.top_k_idxs = top_k_idxs
       
        tr_dset = TopKDataset(self.train_loader_x.dataset.data_source, self.test_loader.dataset.transform, top_k_idxs, top_k_conf, top_k_pseudo)
        tr_loader = DataLoader(tr_dset, batch_size=32, shuffle=True, num_workers=8)
        dino_emb = torch.empty((len(tr_dset), 768))
        label = torch.empty(len(tr_dset), dtype=torch.long)
        label_conf = torch.empty(len(tr_dset))
        ground_truth = torch.empty(len(tr_dset), dtype=torch.long)
        with torch.no_grad():
            for i, batch in enumerate(tr_loader):
                input, lbl = self.parse_batch_train(batch)
                emb = self.model.taal.taal_enc(input)
                dino_emb[i*32:(i+1)*32] = emb
                label[i*32:(i+1)*32] = lbl
                label_conf[i*32:(i+1)*32] = batch['label_conf']
                ground_truth[i*32:(i+1)*32] = batch['ground_truth']
        emb_dict['dino_emb'] = dino_emb
        dict_to_save = {
            'img': dino_emb,
            'label': label,
            'label_conf': label_conf,
            'ground_truth': ground_truth,
        }
        torch.save(dict_to_save, f'embeddings/{self.cfg.DATASET.NAME}_total_emb.pt')
        return DictDataset(dict_to_save)
    
    def train_taal(self, train_loader):
        # setup train taal
        st = time.time()
        optimizer = torch.optim.Adam(self.model.taal.taal_h_module.parameters())
        criterion = nn.CrossEntropyLoss()
        # train the alignment module h
        self.model.taal.eval()
        self.model.taal.taal_h_module.train()
        for epoch in tqdm(range(self.cfg.TAAL_EPOCHS)):
            epoch_time = 0
            for i, batch in enumerate(train_loader):
                input, label = self.parse_batch_train(batch)
                optimizer.zero_grad()
                st = time.time()
                output = self.model.taal.taal_h_module(input)
                epoch_time += time.time()-st
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                if self._writer:
                    self._writer.log({"train/loss_taal": loss.item()})
                # if i % 50 == 0:
                #     print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
            print(f"Epoch: {epoch}, Time: {epoch_time}")
        # print(f"Time taken to train TAAL: {time.time()-st}")        

    def test_taal(self):
        correct = 0
        total = 0
        self.model.taal.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                input, label = self.parse_batch_train(batch)
                output = self.model.taal(input)
                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        if self._writer:
            self._writer.log({"test/accuracy_taal": 100 * correct / total})
        print(f'Accuracy of the network on the {total} test images: {100 * correct / total}')
    
    def after_epoch(self):

        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.test()
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
                self.patience = 15
            else:
                self.patience -= 1
                if self.patience == 0:
                    print("Early stopping the training")
                    sys.exit(0)

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def model_inference(self, input):
        outputs = self.model.eval_clip(input)
        return outputs




