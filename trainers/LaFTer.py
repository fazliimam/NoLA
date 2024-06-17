from torch.nn import Conv2d, Dropout
import math
import os.path as osp
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import tokenize
_tokenizer = _Tokenizer()
from torchvision.models import resnet50, vit_l_16
import torchvision.models as models
import sys
# import sys
sys.path.append("/l/users/sanoojan.baliah/Felix/ALP-RS")
from utils.model_utils import *
from utils.utils import *
_tokenizer = _Tokenizer()
from functools import reduce
from operator import mul
from utils.data_utils import ds_specific_templates

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
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
    
    # Load model


    return model

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

def process_json_files(imagenet_file, dataset_file, output_file, dataset, desc_noise):
    with open(imagenet_file, 'r') as file:
        original_data = json.load(file)

    new_data = {"other": [value for values in original_data.values() for value in values]}

    if dataset=="EuroSAT":
        samples=int(250*desc_noise)
    elif dataset=="Resisc45":
        samples=int(45*50*desc_noise)
    elif dataset=="AID":
        samples=int(30*50*desc_noise)
    elif dataset=="PatternNet":
        samples=int(38*50*desc_noise)
    else:
        # Input samples for other datasets
        samples=int(input("Enter the number of noise samples for the other datasets: "))
    print("Number of noise samples: ", samples)

    sampled_values = random.sample(new_data['other'], samples)
    sampled_data = {"other": sampled_values}

    with open(dataset_file, 'r') as file:
        dataset_data = json.load(file)

    #combine the two dictionaries
    dataset_data.update(sampled_data)
        
    # Save the updated dataset_data to the output file
    with open(output_file, 'w') as file:
        json.dump(dataset_data, file, indent=4)

class Taal(nn.Module):
    def __init__(self, num_classes, templates, device='cuda'):
        super(Taal, self).__init__()

        self.taal_enc =  torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        in_features = self.taal_enc.visual.output_dim
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
      
class LaFTerUFT(nn.Module):

    def __init__(self, model, classes, templates, ds_templates=None, device='cpu', dataset_name=None, txt_cls=None, cfg=None):
        super(LaFTerUFT, self).__init__()

        self.device = device
        self.cfg = cfg
        self.dataset_templates = ds_templates
        self.classes = classes
        self.dataset_name = dataset_name
        self.txt_cls = txt_cls
        patch_size = (16, 16)
        self.model = model.to(device)
        self.templates = templates
        self.backbone_out_size = 512 # todo for ViT-L use 768 - for ViT-B use 512
        self.hidden_size = 768 # todo for ViT-L use 1024 - for ViT-B use 768
        self.num_tokens = 16 # todo all experiments are run with num_tokens = 50
        self.prompt_proj = nn.Identity()
        prompt_dim = self.hidden_size
        
        self.prompt_dropout = Dropout(0.0)

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, self.hidden_size), requires_grad=True)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        self.zs_weights = self.gen_zeroshot_weights()   # ZS classifier weights with all templates
        
        self.txt_features_for_text_cls, self.labels_for_text_cls = self.txt_features_for_text_cls()
        self.text_features = self.txt_features()

        self.taal = Taal(num_classes=len(classes), templates=templates, device=device)

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
            
            # process_json_files('./descriptions/generic/ImageNet.json', f'./descriptions/generic/{self.dataset_name}.json', f'./descriptions/generic/{self.dataset_name}_noised.json', self.dataset_name)

            if self.cfg.desc_noise>0:
                process_json_files('./descriptions/generic/ImageNet.json', f'./descriptions/generic/{self.dataset_name}.json', f'./descriptions/generic/{self.dataset_name}_noised.json', self.dataset_name, self.cfg.desc_noise)
                path_to_file = f'./descriptions/generic/{self.dataset_name}_noised.json'
            elif self.cfg.k_desc>0:
                nos = self.cfg.k_desc
                print(f'******** Using K Descriptions: {nos} *********')
                path_to_file = f'./descriptions/k_descriptions/k_desc/{self.dataset_name}_{nos}.json'
                print(path_to_file)
            else:
                print('******** No Noise Added *********')
                path_to_file = f'./descriptions/generic/{self.dataset_name}.json'
                # path_to_file = f'/l/users/sanoojan.baliah/Felix/dataset_prep/descriptions/{self.dataset_name}_25_promptless_refined.json'
                print(path_to_file)


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
                    class_embeddings = self.model.encode_text(text)  # embed with text encoder # (50, 512) --> embeddings for all 50 texts
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
        
    def gen_zeroshot_weights(self):
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
            
            # process_json_files('./descriptions/generic/ImageNet.json', f'./descriptions/generic/{self.dataset_name}.json', f'./descriptions/generic/{self.dataset_name}_noised.json', self.dataset_name)

            if self.cfg.desc_noise>0:
                process_json_files('./descriptions/generic/ImageNet.json', f'./descriptions/generic/{self.dataset_name}.json', f'./descriptions/generic/{self.dataset_name}_noised.json', self.dataset_name, self.cfg.desc_noise)
                path_to_file = f'./descriptions/generic/{self.dataset_name}_noised.json'
            else:
                print('******** No Noise Added *********')
                path_to_file = f'./descriptions/generic/{self.dataset_name}.json'
                # path_to_file = f'/l/users/sanoojan.baliah/Felix/dataset_prep/descriptions/{self.dataset_name}_25_promptless_refined.json'
                print(path_to_file)


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
                for c in range(len(self.classes)):
                    text = [desc[idx.item()] for idx in (labels_for_descriptions==c).nonzero()]
                    text = tokenize(text).to(self.device)  # tokenize # (50, 77) --> 50 templates/texts from GPT
                    class_embeddings = self.model.encode_text(text)  # embed with text encoder # (50, 512) --> embeddings for all 50 texts
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # L2 norm of the embeddings (dim 2)
                    class_embedding = class_embeddings.mean(dim=0, keepdim=True)
                    class_embedding /= class_embedding.norm()
                    # breakpoint()
                    zeroshot_weights.append(class_embedding)
                zeroshot_weights = torch.stack(zeroshot_weights).to(self.device) # (512, 10) --> 512 embeddings for 10 classes'
                
                # if self.cfg.k_desc>0:
                #     torch.save(zeroshot_weights, f'embeddings/{self.txt_cls}_{self.dataset_name}_K{self.cfg.k_desc}_embeddings.pt')
                # else:
                torch.save(zeroshot_weights, f'embeddings/{self.txt_cls}_{self.dataset_name}_ZSembeddings.pt')

            return zeroshot_weights.squeeze()

        else:
            return None, None
        
    def gen_emb(self):

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
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights

    def image_features(self, images):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def eval_clip(self, x):
        with torch.no_grad():
            img_features_2 = self.incorporate_prompt(x)
            img_features_2 = self.embeddings_after_prompts(img_features_2)
            img_features_adapter = self.adapter(img_features_2)
        return img_features_adapter

    def forward(self, x):
        # only used for 0-shot-eval
        with torch.no_grad():
            img_features = self.image_features(x)
            pseudo_label = img_features @ self.text_features
        return pseudo_label

    def forward_pl_zeroshot(self, x):
        with torch.no_grad():
            if self.cfg.ve_unshared:
                # print('******** Unsharing Vision Encoders *********')
                img_features = self.image_features_frozen_pl(x)
            else:
                img_features = self.image_features(x)

            if self.cfg.desc_emb:
                pseudo_label = img_features @ self.class_desc_emb
            else:
                pseudo_label = img_features @ self.text_features.float()
        return pseudo_label

    def forward_aug_with_prompts(self, x2):
        '''
        :param x1: the clean image (without transforms, for pseudo labels, for teacher)
        :param x2: the transformed image (for student)
        :return: features adapter (cls head), pseudo-labels
        '''
        img_features_2 = self.incorporate_prompt(x2)
        img_features_2 = self.embeddings_after_prompts(img_features_2)
        # breakpoint()
        # img_features_adapter = self.vision_adapter(img_features_2)
        img_features_adapter = self.adapter(img_features_2)
        return img_features_adapter

    def txt_cls_init(self):
        import copy
        self.adapter_pl = copy.deepcopy(self.adapter)
        if self.cfg.classifer_random_weights:
            print('******** Initializing Classifier with Random Weights *********')
            self.adapter.apply(weights_init)
        elif self.cfg.classifier_frozen_weights:
            print('******** Freezing CDTE Weights *********')
            for param in self.adapter.parameters():
                param.requires_grad = False
            
        # if self.cfg.ve_unshared:
        #     self.image_features_frozen = copy.deepcopy(self.model.visual)

    def zs_cls_embs_init(self):
        import copy
        self.adapter_pl = copy.deepcopy(self.adapter)
        if self.cfg.classifer_random_weights:
            print('******** Initializing Classifier with Random Weights *********')
            self.adapter.apply(weights_init)
        elif self.cfg.classifier_frozen_weights:
            print('******** Freezing CDTE Weights *********')
            for param in self.adapter.parameters():
                param.requires_grad = False

        if self.cfg.ve_unshared:
            self.image_features_frozen = copy.deepcopy(self.model.visual)

    def image_features_frozen_pl(self, images):
        with torch.no_grad():
            image_features = self.image_features_frozen(images.type(self.model.dtype))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def forward_normal_for_pl(self, x1):
        '''
        :param x1: the clean image (without transforms, for pseudo labels, for teacher)
        :param x2: the transformed image (for student)
        :return: features adapter (cls head), pseudo-labels
        '''
        with torch.no_grad():
            if self.cfg.ve_unshared:
                 img_features_1 = self.image_features_frozen_pl(x1)   
            else:
                img_features_1 = self.image_features(x1)
            pseudo_label = self.adapter_pl(img_features_1.float()).detach()
        return pseudo_label
    
    def forward_aug_without_prompts(self, x2):
        img_features_2 = self.image_features(x2)
        img_features_adapter = self.adapter(img_features_2)
        return img_features_adapter

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
        return self.model.visual.embeddings_patch(x)

    def embeddings_after_prompts(self, x: torch.tensor):
        return self.model.visual.forward_after_patch_embeddings(x)

    def positional_embeddings_for_text(self, x: torch.tensor):
        return self.model.positional_embeddings(x)

    def embeddings_after_prompts_for_text(self, x: torch.tensor):
        return self.model.embeddings_after_prompting(x)


@TRAINER_REGISTRY.register()
class LaFTer(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        dset = cfg.DATASET.NAME
        print("**********************Dataset: ", dset)
        # breakpoint()
        #TO BE FIXED
        if 'AnnualCrop' in classnames:
            # breakpoint()
            classnames = ['Annual Crop Land', 'Forest', 'Herbaceous Vegetation Land', 'Highway or Road', 'Industrial Buildings', 'Pasture Land', 'Permanent Crop Land', 'Residential Buildings', 'River', 'Sea or Lake']

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        print("Building ZERO-SHOT-MODEL CLIP")

        text_pr = 'a photo of a {}'
        # text_pr = "a satellite photo of {}"
        print(f"Using prompt: {text_pr}")
        self.model = LaFTerUFT(model=clip_model, classes=classnames,
                                          templates=[text_pr], ds_templates = ds_specific_templates[cfg.DATASET.NAME], dataset_name= cfg.DATASET.NAME, txt_cls = cfg.txt_cls, cfg=cfg)
        print("DS Specific Templates: ", ds_specific_templates[cfg.DATASET.NAME])
        self.register_model("adapt", self.model)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        #  freeze clip
        for param in self.model.model.parameters():
            param.requires_grad = False

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

        if isinstance(batch, list):
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
    
    def gen_emb(self):

        # Map unique string values to numerical indices
        unique_groups, group_indices = torch.unique(self.model.labels_for_text_cls, return_inverse=True)

        # Get the total number of unique groups
        total_groups = len(unique_groups)

        # Ensure the number of specified groups is valid
        if len(self.model.classes) > total_groups:
            raise ValueError("The specified number of groups is greater than the total number of unique groups.")

        # Split the indices into the specified number of groups
        group_indices_split = torch.chunk(torch.arange(total_groups), len(self.model.classes))

        # Create a list to store tensors for each group
        group_list = []

        # Iterate over each group and create tensors for the "other" tensor
        for indices in group_indices_split:
            subset_other_tensor = self.model.txt_features_for_text_cls[group_indices == indices[0]]  # Assuming group indices are consistent
            group_list.append(subset_other_tensor.mean(axis=0))

        # Stack the tensors to create a tensor of shape (num_groups, group_dim, emb_dim)
        other_tensor_split = torch.stack(group_list)

        # save the embeddings
        torch.save(other_tensor_split, f'embeddings/{self.model.dataset_name}_avg_text_emb.pt')

        return other_tensor_split.T
    
    def select_top_k(self, k=16):
        text_emb = self.gen_emb()
        # setup for top_k_selection
        total_emb = []
        idxs = []
        labels = []
        for i, batch in tqdm(enumerate(self.train_loader_x)):
            input = torch.stack(batch['img']).to(self.device)
            vision_emb = self.model.image_features(input)
            temp = vision_emb @ text_emb
            total_emb.append(temp)
            idxs.append(batch['index'])
            labels.append(batch['label'])
        total_emb = torch.cat(total_emb)
        idxs = torch.cat(idxs)
        labels = torch.cat(labels)
        return top_k_indices_per_class(total_emb, idxs, labels, k)

    def train_adapter(self, cfg):
        from torch.utils.data import Subset, DataLoader
        train_idxs, _,_,_ = self.select_top_k(k=16)
        train_set = Subset(self.train_loader_x.dataset, train_idxs)
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.taal.parameters(), lr=1e-4)
        # train the adapter
        for epoch in range(cfg.epochs):
            for i, batch in enumerate(train_loader):
                input, label = self.parse_batch_train(batch)
                self.model.taal.train()
                optimizer.zero_grad()
                output = self.model(input)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

    def train_alp_rs(self, cfg):
        from torch.utils.data import Subset, DataLoader
        train_idxs, _,_,_ = self.select_top_k(k=16)
        train_set = Subset(self.train_loader_x.dataset, train_idxs)
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.taal.parameters(), lr=1e-4)
        # train the adapter
        for epoch in range(epochs):
            for i, batch in enumerate(train_loader):
                input, label = self.parse_batch_train(batch)
                self.model.taal.train()
                optimizer.zero_grad()
                output = self.model(input)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")




