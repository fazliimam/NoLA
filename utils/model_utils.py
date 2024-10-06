import torch
import json
import torchvision.transforms as transforms
import PIL.Image as Image
from PIL import ImageFilter
import random
import os
import pandas as pd
import numpy as np
from clip import clip
import torch

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root='all_weights')
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model


def top_k_indices_per_class2(zero_shot_emb, k):
    total_emb = zero_shot_emb['total_emb']
    idxs = zero_shot_emb['idxs']
    gt = zero_shot_emb['labels']
    print(f'Selecting top {k} pseudo labels per class')


    top3_confidences, top3_predictions = torch.topk(total_emb, k=3, dim=1)
    df = pd.DataFrame({
        'idxs': idxs.tolist(),
        'labels': gt.tolist(),
        'pred1': top3_predictions[:, 0].tolist(),
        'pred2': top3_predictions[:, 1].tolist(),
        'pred3': top3_predictions[:, 2].tolist(),
        'prob1': top3_confidences[:, 0].tolist(),
        'prob2': top3_confidences[:, 1].tolist(),
        'prob3': top3_confidences[:, 2].tolist(),
    })

    df['correct'] = (df['labels'] == df['pred1']).astype(int)
    print('Correct predictions percentage: ', df['correct'].mean())
    pseudo_df = pd.DataFrame()


    for pred_label in set(df.labels):
        sub_label_df = df.loc[(df.pred1 == pred_label)]
        sub_label_df = sub_label_df.sort_values('prob1', ascending=False).iloc[0:k]

        if len(sub_label_df) == 0:
            sub_label_df = df.loc[(df.pred2 == pred_label)]
            sub_label_df = sub_label_df.sort_values('prob2', ascending=False).iloc[0:k]
            sub_label_df['pred1'] = sub_label_df['pred2']
            print(f'For label {pred_label}, {len(sub_label_df)} rows selected')
            if len(sub_label_df) == 0:
                sub_label_df = df.loc[(df.pred3 == pred_label)]
                sub_label_df = sub_label_df.sort_values('prob3', ascending=False).iloc[0:k]
                sub_label_df['pred1'] = sub_label_df['pred3']
                print(f'For label {pred_label}, {len(sub_label_df)} rows selected')
                if len(sub_label_df) == 0:
                    raise NotImplementedError
        correct_acc = (sub_label_df['labels'] == sub_label_df['pred1']).sum() / len(sub_label_df)
        print(f'acc per class {pred_label}: {correct_acc}')
        pseudo_df = pd.concat((pseudo_df, sub_label_df))    

    pseudo_df = pseudo_df.drop_duplicates(subset=['idxs'])
    print('Selected pseudo label accuracy: ', (pseudo_df['correct']).mean())

    top_k_original_indices = torch.tensor(pseudo_df['idxs'].values)
    top_k_values = torch.tensor(pseudo_df['prob1'].values)
    top_k_pseudo_labels = torch.tensor(pseudo_df['pred1'].values)

    
    return top_k_original_indices.flatten(), top_k_values.flatten(), top_k_pseudo_labels.flatten()


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, random_transform):
        self.base_transform = base_transform  # from clip
        self.random_tranform = random_transform  # random transforms (currently from simsiam)

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.random_tranform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_random_transform(ndim):
    normalize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                         (0.26862954, 0.26130258, 0.27577711))])

    blur = GaussianBlur()
    print('Using Default transforms + CJ')

    return transforms.Compose([
        transforms.RandomResizedCrop(ndim, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        blur,
        normalize
    ])


te_transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    lambda image: image.convert("RGB"),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

transform_default_clip_weakly_aug = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    lambda image: image.convert("RGB"),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

tr_transforms = TwoCropsTransform(te_transform, get_random_transform(224))

def gen_labels_with_templates(classes, descriptions):
    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        if '_' in classname:
            classname = classname.replace('_', ' ')

        for descp in descriptions:
            descp = descp.format(classname)
            desc_.append(descp)
            labels.append(i)
    return desc_, labels


def gen_labels_with_captions(classes, folder_path, args):
    if args.dataset == 'imagenet':
        desc_ = []
        labels = []
        cls_name_dict = {}
        with open(os.path.join(folder_path, 'imagenet_class_index.json')) as f:
            cls_idx = json.load(f)
        for k, v in cls_idx.items():
            cls_name_dict[v[0]] = v[1]
        for i, classname in enumerate(cls_name_dict.keys()):
            with open(os.path.join(folder_path, f'{classname}.txt'), 'r') as f:
                for line in f:
                    desc_.append(line.split(" ", 1)[1].replace("\n", "").lower())
                    labels.append(i)

        return desc_, labels

    desc_ = []
    labels = []
    classes_to_care = ['aquarium fish', 'lawn mower', 'maple tree', 'oak tree', 'pickup truck', 'pine tree',
                       'sweet pepper', 'willow tree']
    for i, classname in enumerate(classes):
        if classname in classes_to_care:
            split_ = 2
        else:
            split_ = 1
        with open(os.path.join(folder_path, f'{classname}.txt'), 'r') as f:
            for line in f:
                desc_.append(line.split(" ", split_)[split_].replace("\n", "").lower())
                labels.append(i)
    return desc_, labels


def gen_labels_with_captions_blip_2(classes, folder_path, args):
    with open(folder_path) as f:
        lines = f.readlines()
    desc_ = []
    labels = []

    classes_to_care = ['aquarium fish', 'lawn mower', 'maple tree', 'oak tree', 'pickup truck', 'pine tree',
                       'sweet pepper', 'willow tree']

    for i, c in enumerate(classes):
        if c in classes_to_care:
            split_ = 2
        else:
            split_ = 1
        for l in lines:
            if l.strip().split(' ')[0].split('/')[-2] == c:
                labels.append(i)
                desc_.append(l.strip().split(' ', split_)[split_].replace("\n", "").lower())
                print(l.strip().split(' ', split_)[split_].replace("\n", "").lower())
    return desc_, labels


def gen_labels_with_classes(classes, descriptions):
    # for direct class -> class
    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        desc_.append(classname)
        labels.append(i)
    return desc_, labels


def gen_labels_with_classes_and_simple_template(classes, descriptions):
    # for direct class -> simple template
    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        descp = f'a photo of a {classname}'
        desc_.append(descp)
        labels.append(i)
    return desc_, labels


def gen_labels_with_synonyms(classes, folder_path, args):
    with open(os.path.join(folder_path, f'{args.dataset}_cleaned.json')) as f:
        cls_idx = json.load(f)
    desc_ = []
    labels = []
    for i, (k, v) in enumerate(cls_idx.items()):
        classes = cls_idx[k].split(',')
        for j, classname in enumerate(classes):
            desc_.append('a photo of a ' + classname + '.')
            labels.append(i)
    return desc_, labels


def gen_labels_with_descrptions(classes, descriptions):
    desc_ = []
    labels = []
    if len(classes)==397:
        classes = descriptions.keys() # uncomment this for sun397
    for i, classname in enumerate(classes):
        for desc in descriptions[classname]:
            desc_.append(desc)
            labels.append(i)
    return desc_, labels

def process_json(path_to_file, classes):
    with open(path_to_file) as f:
        gpt3_prompts = json.load(f)
    desc, labels_for_descriptions = gen_labels_with_descrptions(classes, descriptions=gpt3_prompts)
    return desc, labels_for_descriptions

def gen_labels_with_expanded_labels_imagenet(folder, args):
    if args.dataset != 'imagenet':
        raise ValueError('Only for imagenet')
    with open(os.path.join(folder, 'imagenet_expanded_labels.txt')) as f:
        data = f.readlines()
    exp_cls = list()
    for i, d in enumerate(data):
        exp_cls.append(d.split(' ', 1)[1].split(' ', 1)[1].replace('\n', '').replace('\'', '').replace('\"', ''))
    desc_ = []
    labels = []
    for i, c in enumerate(exp_cls):
        cls = c.split(',')
        for j, c_ in enumerate(cls):
            if c_ == '':
                continue
            desc_.append('a photo of a ' + c_.replace(' ', ''))
            labels.append(i)
    return desc_, labels

def gen_labels_with_descrptions_and_clsname(classes, descriptions):
    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        for desc in descriptions[classname]:
            desc_.append(classname + ': ' + desc)
            labels.append(i)

    return desc_, labels