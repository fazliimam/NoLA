from torch.nn import Dropout
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import math
import os.path as osp
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import read_image
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import tokenize
_tokenizer = _Tokenizer()
from torch.utils.data import Dataset, DataLoader
import time
import sys
# import sys
sys.path.append("../")
from utils.model_utils import *
from utils.utils import *
_tokenizer = _Tokenizer()
from functools import reduce
from operator import mul
from utils.data_utils import ds_specific_templates



def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root='all_weights')
    # model_path="all_weights/RemoteCLIP-ViT-B-32.pt"
    # model_path ="all_weights/RS5M_ViT-B-32.pt"1
    print("_______________________________")
    print(model_path)
    print("_______________________________")
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    
    return model



class lossmeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


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
    
    
# class TopKDataset1(Dataset):
#     def __init__(self, data_list):
#         self.data_list = data_list
#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         return self.data_list[idx]
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
        self.taal_adt = AdapterMLP(num_classes=num_classes, input_size=in_features, hidden_size=256)

    def forward(self, x):
        with torch.no_grad():
            op = self.taal_enc(x)
        op = self.taal_adt(op)
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
      
class ALP_RS(nn.Module):

    def __init__(self, clip_model, classes, templates, ds_templates=None, cfg=None):
        super(ALP_RS, self).__init__()

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
        self.taal = Taal(num_classes=len(classes), templates=templates, device=cfg.DEVICE).to(self.device)

        self.adapter = nn.Sequential(nn.Linear(int(self.backbone_out_size), len(classes), bias=False)).to(self.device)

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, self.hidden_size), requires_grad=True)
        patch_size = (16, 16)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        # register the prompt embeddings as named parameters
        

        self.txt_features_for_text_cls, self.labels_for_text_cls = self.txt_features_for_text_cls()
        self.avg_text_emb = self.gen_emb2()  # Average text embeddings for each class
        # self.text_features = self.txt_features()

    def txt_features_for_text_cls(self):

        if self.txt_cls== 'cls_only':
            gpt3_prompts = None
            desc, labels_for_descriptions = gen_labels_with_classes(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == 'templates_only':
            gpt3_prompts = self.templates
            desc, labels_for_descriptions = gen_labels_with_templates(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == 'lafter':
            # generic prompts + templates

            if self.dataset_name not in lafter_datasets:
                raise ValueError('Invalid dataset name for LaFTer')
            

            path_to_file = f'./descriptions/generic/{self.dataset_name}.json'
            
            with open(path_to_file) as f:
                gpt3_prompts = json.load(f)
            desc, labels_for_descriptions = gen_labels_with_descrptions(self.classes, descriptions=gpt3_prompts)

            templates, labels_for_templates = gen_labels_with_templates(self.classes, descriptions=self.dataset_templates)

            desc += templates
            labels_for_descriptions += labels_for_templates

        elif self.txt_cls == 'zero_shot':
            pass

        else:
            raise ValueError('Invalid txt_cls argument')


        if self.txt_cls in ['cls_only', 'templates_only', 'lafter']:

            Path(f'embeddings').mkdir(parents=True, exist_ok=True)

            if os.path.isfile(f'embeddings/{self.txt_cls}_{self.dataset_name}_embeddings.pt'):
                zeroshot_weights = torch.load(f'embeddings/{self.txt_cls}_{self.dataset_name}_embeddings.pt')
                print('******** Loaded Already Saved Embeddings *********')
                labels_for_descriptions = torch.tensor(labels_for_descriptions).to(self.device)

            else:
                print('******** No Embeddings Found --- Saving New Embeddings *********')

            labels_for_descriptions = torch.tensor(labels_for_descriptions).to(self.device)

            zeroshot_weights = []
            with torch.no_grad():
                for classname in tqdm(desc):
                    text = tokenize(classname).to(self.device)  # tokenize # (50, 77) --> 50 templates/texts from GPT
                    class_embeddings = self.clip_model.encode_text(text)  # embed with text encoder # (50, 512) --> embeddings for all 50 texts
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # L2 norm of the embeddings (dim 2)
                    zeroshot_weights.append(class_embeddings)
                zeroshot_weights = torch.stack(zeroshot_weights).to(self.device)  # (512, 10) --> 512 embeddings for 10 classes'
                
                # if self.cfg.k_desc>0:
                #     torch.save(zeroshot_weights, f'embeddings/{self.txt_cls}_{self.dataset_name}_K{self.cfg.k_desc}_embeddings.pt')
                # else:
                torch.save(zeroshot_weights, f'embeddings/{self.txt_cls}_{self.dataset_name}_embeddings.pt')

            return zeroshot_weights.squeeze(), labels_for_descriptions

        else:
            return None, None
        
    def gen_emb(self):
        if os.path.isfile(f'embeddings/{self.dataset_name}_avg_text_emb.pt'):
            print('******** Loading Already Saved Averaged Text Embeddings *********')
            return torch.load(f'embeddings/{self.dataset_name}_avg_text_emb.pt').T
        # Map unique string values to numerical indices
        unique_groups, group_indices = torch.unique(self.labels_for_text_cls, return_inverse=True)

        # Get the total number of unique groups
        total_groups = len(unique_groups)

        # Ensure the number of specified groups is valid
        if len(self.classes) > total_groups:
            raise ValueError("The specified number of groups is greater than the total number of unique groups.")

        # Split the indices into the specified number of groups
        group_indices_split = torch.chunk(torch.arange(total_groups), len(self.classes))

        # Create a list to store tensors for each group
        group_list = []

        # Iterate over each group and create tensors for the "other" tensor
        for indices in group_indices_split:
            subset_other_tensor = self.txt_features_for_text_cls[group_indices == indices[0]]  # Assuming group indices are consistent
            group_list.append(subset_other_tensor.mean(axis=0))

        # Stack the tensors to create a tensor of shape (num_groups, group_dim, emb_dim)
        other_tensor_split = torch.stack(group_list)

        # save the embeddings
        # torch.save(other_tensor_split, f'embeddings/{self.dataset_name}_avg_text_emb.pt')
        breakpoint()
        return other_tensor_split.T


    def gen_emb2(self):
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

    def txt_features(self):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(self.classes):
                texts = [template.format(classname) for template in self.templates]  # format with class
                texts = tokenize(texts).to(self.device)  # tokenize
                class_embeddings = self.clip_model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
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
            zs_emb = img_features_2 @ self.avg_text_emb
        return zs_emb

    def forward(self, x):
        # only used for 0-shot-eval
        with torch.no_grad():
            img_features = self.image_features(x)
            pseudo_label = img_features.to(self.avg_text_emb.device) @ self.avg_text_emb
        return pseudo_label

    def forward_aug_with_prompts(self, x2):
        '''
        :param x2: the transformed image (for student)
        :return: features adapter (cls head), pseudo-labels
        '''
        img_features_2 = self.incorporate_prompt(x2)
        img_features_2 = self.embeddings_after_prompts(img_features_2)
        img_features_adapter = self.adapter(img_features_2)
        return img_features_adapter

    def forward_with_text_desc(self, image, text_desc_emb):
        image_features = self.image_features(image)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # text_desc_emb = text_desc_emb.half()
        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_desc_emb
        logits_per_text = logit_scale * text_desc_emb.t() @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    
    def forward_taal(self,x):
        return self.taal(x)

    def forward_aug_without_prompts(self, x2):
        img_features_2 = self.image_features(x2)
        zs_emb = self.avg_text_emb @ img_features_2
        return zs_emb

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
class ALP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def setup_optim(self):
        params = []
        for k, v in self.model.named_parameters():
            if 'prompt_embeddings' in k:
                v.requires_grad = True
                params.append((k,v))
            
            if 'ln' in k and 'visual' in k:
                v.requires_grad = True
                params.append((k,v))
            else:
                v.requires_grad = False

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
        classnames = self.dm.dataset.classnames
        dset = cfg.DATASET.NAME
        print("**********************Dataset: ", dset)
        #TO BE FIXED
        if 'AnnualCrop' in classnames:
            # breakpoint()
            classnames = ['Annual Crop Land', 'Forest', 'Herbaceous Vegetation Land', 'Highway or Road', 'Industrial Buildings', 'Pasture Land', 'Permanent Crop Land', 'Residential Buildings', 'River', 'Sea or Lake']

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg.MODEL.BACKBONE.NAME)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        print("Building ZERO-SHOT-MODEL CLIP")

        text_pr = 'a photo of a {}'
        # text_pr = "a satellite photo of {}"
        print(f"Using prompt: {text_pr}")
        self.model = ALP_RS(clip_model=clip_model, classes=classnames,
                                          templates=[text_pr], ds_templates = ds_specific_templates[cfg.DATASET.NAME], cfg=cfg)
        
        self.register_model("adapt", self.model)

        for k,v in self.model.clip_model.named_parameters():
            v.requires_grad = False

        # self.optim, self.scheduler, self.criterion = self.setup_optim() 

        # self.register_model("adapt", self.model, self.optim, self.scheduler)

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

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    # def before_train(self):

    #     # Initialize summary writer
    #     writer_dir = osp.join(self.output_dir, "tensorboard")
    #     os.makedirs(writer_dir, exist_ok=True)
    #     self.init_writer(writer_dir)

    #     # if self.cfg.STAGE == 'train_alp':
    #     #     return
        
    #     # Remember the starting time (for computing the elapsed time)
    #     self.time_start = time.time()

    #     # get top_k dataset
    #     top_k_dataset = self.select_top_k(k=16)

    #     # train taal using top_k dataset
    #     train_loader = DataLoader(top_k_dataset, batch_size=32, shuffle=True, num_workers=4)

    #     self.train_taal(self.cfg, train_loader)
    #     self.test_taal(self.cfg)    

    #     self.model = setup_train_alp(self.model)

    def forward_backward(self, batch_x):

        input_x, label_x = self.parse_batch_train(batch_x)

        pl = self.model.taal(input_x[0])
        pseudo_label = F.softmax(pl, dim=-1)  # / 0.04
        pseudo_label = pseudo_label.argmax(dim=1, keepdim=True)
        pseudo_label = pseudo_label.flatten()

        output_x = self.model.forward_aug_with_prompts(input_x[1].float())
        loss_x = self.criterion(output_x.squeeze(), pseudo_label)

        self.model_backward_and_update(loss_x, names="adapt")
        self.update_lr(names="adapt")
        # self.model_backward_and_update(loss_x)
        # self.update_lr(names="adapt")

        loss_summary = {
            "loss_x": loss_x.item(),
        }
        return loss_summary

    def build_topk_dataset(self, top_k_idxs, top_k_conf, top_k_pseudo):
        data = []
        st = time.time()
        print("Building TopK Dataset")
        for idx, conf, pseudo in zip(top_k_idxs, top_k_conf, top_k_pseudo):
            data.append({
                'img': self.train_loader_x.dataset[idx]['img'],
                'label': pseudo,
                'label_conf': conf,
                'ground_labels': self.train_loader_x.dataset[idx]['label'],
            })
        print(f"Time taken to build TopK Dataset: {time.time()-st}")
        return TopKDataset1(data)
    

    def select_top_k(self, k):
        # if file exists, load the embeddings
        if os.path.isfile(f'embeddings/{self.cfg.DATASET.NAME}_total_emb.pt'):
            print('******** Loading Already Saved Averaged Text Embeddings *********')
            total_emb = torch.load(f'embeddings/{self.cfg.DATASET.NAME}_total_emb.pt')
            # top_k_idxs, top_k_conf, top_k_pseudo = top_k_indices_per_class2(total_emb, k)
            # self.top_k_idxs = top_k_idxs
            return DictDataset(total_emb)
            # return TopKDataset(self.train_loader_x.dataset.data_source, self.val_loader.dataset.transform, top_k_idxs, top_k_conf, top_k_pseudo)
        # file_path = f'embeddings/{self.cfg.DATASET.NAME}_avg_text_emb.pt'
        print("******** Generating Embeddings ********")
        text_emb =  self.model.avg_text_emb

        # Check if the file exists
        # if os.path.exists(file_path):
        #     # Load the embeddings from the file if it exists
        #     text_emb = torch.load(file_path)
        # else:
        #     # Generate the embeddings using the model's method if the file does not exist
        #     text_emb = self.model.gen_emb()
            
        # setup for top_k_selection
        total_emb = []
        idxs = []
        labels = []
        impaths = []
        train_loader = DataLoader(self.train_loader_x.dataset, batch_size=128, shuffle=False, num_workers=4)
        for batch in tqdm(train_loader):
            input = torch.stack(batch['img']).to(self.device)
            # vision_emb = self.model.image_features(input[0])
            # temp = vision_emb @ text_emb
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
        # forward pass on the taal encoder and save the embedings
        # tr_dset = TopKDataset(self.train_loader_x.dataset.data_source, self.val_loader.dataset.transform, top_k_idxs, top_k_conf, top_k_pseudo)
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
        # return TopKDataset(self.train_loader_x.dataset, top_k_idxs, top_k_conf, top_k_pseudo)
        # dataset class from dict
        return DictDataset(dict_to_save)
    
    def train_taal(self, train_loader):
        # setup train taal
        st = time.time()

        optimizer = setup_train_taal(self.model.taal)

        criterion = nn.CrossEntropyLoss()
        # train the adapter
        self.model.taal.train()
        for epoch in tqdm(range(self.cfg.TAAL_EPOCHS)):
            epoch_time = 0
            for i, batch in enumerate(train_loader):
                input, label = self.parse_batch_train(batch)
                optimizer.zero_grad()
                st = time.time()
                output = self.model.taal.taal_adt(input)
                epoch_time += time.time()-st
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                if self._writer:
                    self._writer.add_scalar("train/loss_taal", loss.item(), epoch * len(train_loader) + i)
                # if i % 50 == 0:
                #     print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
            print(f"Epoch: {epoch}, Time: {epoch_time}")
        # print(f"Time taken to train TAAL: {time.time()-st}")        

    def test_taal(self, cfg):
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
            self._writer.add_scalar("test/accuracy_taal", 100 * correct / total, 0)
        print(f'Accuracy of the network on the {total} test images: {100 * correct / total}')

    def model_inference(self, input):
        return self.model.eval_clip(input)

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
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
            return curr_result

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

        return None
    
    def train(self, start_epoch=0, max_epoch=50, cfg=None):
        # print("************")
        # print("** Config **")
        # print("************")
        # print(cfg)


        """Generic training loop with early stopping based on patience."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        # writer_dir = osp.join(self.output_dir, "tensorboard")
        # os.makedirs(writer_dir, exist_ok=True)
        # self.init_writer(writer_dir)
        self.time_start = time.time()

        nos_train = len(self.train_loader_x.dataset)
        nos_classes = len(self.dm.dataset.classnames)
        k=find_k(nos_train, nos_classes)
        print("************")
        print("** Setting k **", k)
        print("************")
        top_k_dataset = self.select_top_k(k)

        train_loader = DataLoader(top_k_dataset, batch_size=32, shuffle=True, num_workers=4)

        self.train_taal(train_loader)
        self.test_taal(self.cfg)    


        # self.model = setup_train_alp(self.model)
        optimizer, scheduler, criteria = setup_train_nola(self.model, lrate=self.cfg.OPTIM.LR)
        tr_loader, val_loader, te_loader = self.train_loader_x, self.val_loader, self.test_loader

        
        # print('------------------ Learnable Parameters in NoLA ------------------')
        # for key, value in self.model.named_parameters():
        #     if value.requires_grad:
        #         print("\t{}, {}, {}".format(key, value.numel(), value.shape))
        # print('----------------------------------------------------------')

        # Initialize early stopping parameters
        early_stopping_counter = 0
        early_stopping_threshold = 15
        best_acc = 0.0

        if self.model.prompt_embeddings.requires_grad:
            print("Prompt embeddings are trainable.")

        for epoch in range(start_epoch, max_epoch):

            for i, batch in tqdm(enumerate(tr_loader), total=len(tr_loader)):
                input = batch["img"]
                input = torch.stack(input)  # two views from dataloader
                input = input.to(self.model.device)

                optimizer.zero_grad()
                with torch.no_grad():
                    pl = self.model.taal(input[0])
                    pseudo_label = F.softmax(pl, dim=-1)  # / 0.04
                    pseudo_label = pseudo_label.argmax(dim=1, keepdim=True)
                    pseudo_label = pseudo_label.flatten()

                output_x = self.model.forward_aug_with_prompts(input[1].float())
                loss = criteria(output_x.squeeze(), pseudo_label)
                    
                loss.backward()
                optimizer.step()
            scheduler.step()
            try:
                val_acc = test_prompting(val_loader, self.model)
            except:
                val_acc = test_prompting(te_loader, self.model)

            print(f'Validation Accuracy: {val_acc} at epoch {epoch}')

            if val_acc>best_acc:
                best_acc=val_acc
                early_stopping_counter = 0
                best_test_acc = test_prompting(te_loader, self.model)
                print("Best Epoch ", epoch)
                print("Best Val acc", val_acc)
                print("Test acc ", best_test_acc)

            else:
                early_stopping_counter += 1

            if early_stopping_counter == early_stopping_threshold:
                print(f'Early stopping at epoch {epoch} due to no improvement in validation accuracy.')
                break

            print('------------')
   

        # self.after_train()





