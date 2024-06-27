from torch.nn import Dropout
from torch.utils.data import DataLoader
import math
import os.path as osp
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import tokenize
_tokenizer = _Tokenizer()
from torch.utils.data import Dataset, DataLoader

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



class TopKDataset(Dataset):
    def __init__(self, data, top_k_idxs, top_k_conf, top_k_pseudo):
        self.data = data
        self.top_k_idxs = top_k_idxs
        self.top_k_conf = top_k_conf
        self.top_k_pseudo = top_k_pseudo

    def __len__(self):
        return len(self.top_k_idxs)

    def __getitem__(self, idx):
        new_idx = self.top_k_idxs[idx]
        return {
            'img': self.data[new_idx]['img'],
            'label': torch.tensor(self.top_k_pseudo[idx], dtype=torch.long),
            'label_conf': torch.tensor(self.top_k_conf[idx], dtype=torch.float32),
            'ground_labels': torch.tensor(self.data[new_idx]['label'], dtype=torch.long),
        }
    
class TopKDataset1(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
class Taal(nn.Module):
    def __init__(self, num_classes, templates, device='cuda'):
        super(Taal, self).__init__()

        self.taal_enc =  torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        in_features = self.taal_enc.cls_token.shape[-1]
        self.taal_adt = AdapterMLP(num_classes=num_classes, input_size=in_features, hidden_size=256)

    def forward(self, x):
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

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, self.hidden_size), requires_grad=True).to(self.device)
        patch_size = (16, 16)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        self.txt_features_for_text_cls, self.labels_for_text_cls = self.txt_features_for_text_cls()
        self.avg_text_emb = self.gen_emb()  # Average text embeddings for each class
        self.text_features = self.txt_features()

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
        torch.save(other_tensor_split, f'embeddings/{self.dataset_name}_avg_text_emb.pt')

        return other_tensor_split.T

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
            zs_emb = self.avg_text_emb @ img_features_2
        return zs_emb

    def forward(self, x):
        # only used for 0-shot-eval
        with torch.no_grad():
            img_features = self.image_features(x)
            pseudo_label = img_features @ self.avg_text_emb
        return pseudo_label

    def forward_aug_with_prompts(self, x2):
        '''
        :param x2: the transformed image (for student)
        :return: features adapter (cls head), pseudo-labels
        '''
        img_features_2 = self.incorporate_prompt(x2)
        img_features_2 = self.embeddings_after_prompts(img_features_2)
        zs_emb = self.avg_text_emb.T @ img_features_2.T
        return zs_emb

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
        for k, v in self.model.clip_model.named_parameters():
            if 'ln' in k:
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
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, None, 0.60)
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

    def parse_batch_train(self, batch):

        if isinstance(batch['img'], list):
            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(self.device)
        else:
            input = batch['img']
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
    
    def before_train(self):

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        self.init_writer(writer_dir)

        # if self.cfg.STAGE == 'train_alp':
        #     return
        
        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

        # get top_k dataset
        top_k_dataset = self.select_top_k(k=16)

        # train taal using top_k dataset
        train_loader = DataLoader(top_k_dataset, batch_size=8192, shuffle=True, num_workers=4)

        self.train_taal(self.cfg, train_loader)
        self.test_taal(self.cfg)

        self.model = setup_train_alp(self.model)

    def forward_backward(self, batch_x):

        input_x, label_x = self.parse_batch_train(batch_x)

        out_psuedo = self.model.taal(input_x[0])
        output_x = self.model.forward_aug_with_prompts(input_x[1].float())
        loss_x = self.criterion(output_x.squeeze(), out_psuedo)

        self.model_backward_and_update(loss_x)

        loss_summary = {
            "loss_x": loss_x.item(),
        }
        return loss_summary

    def build_topk_dataset(self, top_k_idxs, top_k_conf, top_k_pseudo):
        data = []
        for idx, conf, pseudo in zip(top_k_idxs, top_k_conf, top_k_pseudo):
            data.append({
                'img': self.train_loader_x.dataset[idx]['img'],
                'label': pseudo,
                'label_conf': conf,
                'ground_labels': self.train_loader_x.dataset[idx]['label'],
            })
        
        return TopKDataset1(data)

    def select_top_k(self, k=16):
        # if file exists, load the embeddings
        if os.path.isfile(f'embeddings/{self.cfg.DATASET.NAME}_total_emb.pt'):
            print('******** Loading Already Saved Averaged Text Embeddings *********')
            total_emb = torch.load(f'embeddings/{self.cfg.DATASET.NAME}_total_emb.pt')
            top_k_idxs, top_k_conf, top_k_pseudo = top_k_indices_per_class(total_emb, k)
            self.top_k_idxs = top_k_idxs

            # return TopKDataset(self.train_loader_x.dataset, top_k_idxs.cpu().numpy(), top_k_conf.cpu().numpy(), top_k_pseudo.cpu().numpy())
            return self.build_topk_dataset(top_k_idxs.cpu().numpy(), top_k_conf.cpu().numpy(), top_k_pseudo.cpu().numpy())
        
        # file_path = f'embeddings/{self.cfg.DATASET.NAME}_avg_text_emb.pt'

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
        train_loader = DataLoader(self.train_loader_x.dataset, batch_size=16, shuffle=False, num_workers=4)
        self.model.eval()
        for batch in tqdm(train_loader):
            input = torch.stack(batch['img']).to(self.device)
            vision_emb = self.model.image_features(input[0])
            temp = vision_emb @ text_emb
            total_emb.append(temp)
            idxs.append(batch['index'])
            impaths.append(batch['impath'])
            labels.append(batch['label'])

        emb_dict = {
                'total_emb': torch.cat(total_emb),
                'idxs': torch.cat(idxs),
                'labels':  torch.cat(labels),
            }
        
        torch.save(emb_dict, f'embeddings/{self.cfg.DATASET.NAME}_total_emb.pt')
        top_k_idxs, top_k_conf, top_k_pseudo = top_k_indices_per_class(emb_dict, k)
        self.top_k_idxs = top_k_idxs

        # return TopKDataset(self.train_loader_x.dataset, top_k_idxs, top_k_conf, top_k_pseudo)
        return self.build_topk_dataset(top_k_idxs.cpu().numpy(), top_k_conf.cpu().numpy(), top_k_pseudo.cpu().numpy())
    
    def train_taal(self, cfg, train_loader):
        # setup train taal
        optimizer = setup_train_taal(self.model.taal)

        criterion = nn.CrossEntropyLoss()
        # train the adapter
        self.model.taal.train()
        for epoch in tqdm(range(cfg.TAAL_EPOCHS)):
            for i, batch in enumerate(train_loader):
                input, label = self.parse_batch_train(batch)
                optimizer.zero_grad()
                output = self.model.taal(input[0])
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                if self._writer:
                    self._writer.add_scalar("train/loss_taal", loss.item(), epoch * len(train_loader) + i)
                # if i % 10 == 0:
                #     print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

    def test_taal(self, cfg):
        correct = 0
        total = 0
        self.model.taal.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                input, label = self.parse_batch_train(batch)
                output = self.model.taal(input)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        if self._writer:
            self._writer.add_scalar("test/accuracy_taal", 100 * correct / total, 0)
        print(f'Accuracy of the network on the {total} test images: {100 * correct / total}')




