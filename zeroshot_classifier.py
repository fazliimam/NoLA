import torch
from tqdm import tqdm
import torch.nn as nn
from clip.clip import tokenize
import json
import os
from clip import clip
from trainers.alp import *

print("Life is good")

ds_specific_templates = {
    'DescribableTextures': [
        'a photo of a {} texture.',
        'a photo of a {} pattern.',
        'a photo of a {} thing.',
        'a photo of a {} object.',
        'a photo of the {} texture.',
        'a photo of the {} pattern.',
        'a photo of the {} thing.',
        'a photo of the {} object.',
    ],
    'EuroSAT':[
        'a satellite photo of {}.',
        'a satellite photo of a {}.',
        'a satellite photo of the {}.',
        'a centered satellite photo of {}.',
        'This is an aerial image of a {}.',
        'a aerial image of {}.',
    ],
    'RESISC45':[
        'a satellite photo of {}.',
        'a satellite photo of a {}.',
        'a satellite photo of the {}.',
        'a centered satellite photo of {}.',
        'This is an aerial image of a {}.',
        'a aerial image of {}.',
    ],
    'Optimal31':[
        'a satellite photo of {}.',
        'a satellite photo of a {}.',
        'a satellite photo of the {}.',
        'a centered satellite photo of {}.',
        'This is an aerial image of a {}.',
        'a aerial image of {}.',
    ],
    'AID':[
        'a satellite photo of {}.',
        'a satellite photo of a {}.',
        'a satellite photo of the {}.',
        'a centered satellite photo of {}.',
        'This is an aerial image of a {}.',
        'a aerial image of {}.',
    ],
    'WHURS19': [
        'a satellite photo of {}.',
        'a satellite photo of a {}.',
        'a satellite photo of the {}.',
        'a centered satellite photo of {}.',
        'This is an aerial image of a {}.',
        'a aerial image of {}.',
    ],
    'MLRSNet':[
        'a satellite photo of {}.',
        'a satellite photo of a {}.',
        'a satellite photo of the {}.',
        'a centered satellite photo of {}.',
        'This is an aerial image of a {}.',
        'a aerial image of {}.',
    ],
    'UCM':[
        'a satellite photo of {}.',
        'a satellite photo of a {}.',
        'a satellite photo of the {}.',
        'a centered satellite photo of {}.',
        'This is an aerial image of a {}.',
        'a aerial image of {}.',
        # 'a land use image of {}.',
        # 'This is an aerial image of a {}.',
        # 'This is an aerial photo of a {}.',
    ],
    'PatternNet':[
        'a satellite photo of {}.',
        'a satellite photo of a {}.',
        'a satellite photo of the {}.',
        'a centered satellite photo of {}.',
        'This is an aerial image of a {}.',
        'a aerial image of {}.',

    ],
    'OxfordFlowers': [
        'a photo of a {}, a type of flower.',
    ],
    'SUN397': [
        'a photo of a {}.',
        'a photo of the {}.',
    ],
    'UCF101': [
        'a photo of a person {}.',
        'a video of a person {}.',
        'a example of a person {}.',
        'a demonstration of a person {}.',
        'a photo of the person {}.',
        'a video of the person {}.',
        'a example of the person {}.',
        'a demonstration of the person {}.',
        'a photo of a person using {}.',
        'a video of a person using {}.',
        'a example of a person using {}.',
        'a demonstration of a person using {}.',
        'a photo of the person using {}.',
        'a video of the person using {}.',
        'a example of the person using {}.',
        'a demonstration of the person using {}.',
        'a photo of a person doing {}.',
        'a video of a person doing {}.',
        'a example of a person doing {}.',
        'a demonstration of a person doing {}.',
        'a photo of the person doing {}.',
        'a video of the person doing {}.',
        'a example of the person doing {}.',
        'a demonstration of the person doing {}.',
        'a photo of a person during {}.',
        'a video of a person during {}.',
        'a example of a person during {}.',
        'a demonstration of a person during {}.',
        'a photo of the person during {}.',
        'a video of the person during {}.',
        'a example of the person during {}.',
        'a demonstration of the person during {}.',
        'a photo of a person performing {}.',
        'a video of a person performing {}.',
        'a example of a person performing {}.',
        'a demonstration of a person performing {}.',
        'a photo of the person performing {}.',
        'a video of the person performing {}.',
        'a example of the person performing {}.',
        'a demonstration of the person performing {}.',
        'a photo of a person practicing {}.',
        'a video of a person practicing {}.',
        'a example of a person practicing {}.',
        'a demonstration of a person practicing {}.',
        'a photo of the person practicing {}.',
        'a video of the person practicing {}.',
        'a example of the person practicing {}.',
        'a demonstration of the person practicing {}.',
    ],
    'ImageNetR': [
        'a bad photo of the {}.',
        'a {} in a video game.',
        'a origami {}.',
        'a photo of the small {}.',
        'art of the {}.',
        'a photo of the large {}.',
        'itap of a {}.',
    ],
    'ImageNetSketch': [
        'a bad photo of the {}.',
        'a {} in a video game.',
        'a origami {}.',
        'a photo of the small {}.',
        'art of the {}.',
        'a photo of the large {}.',
        'itap of a {}.',
    ],
    'ImageNetA': [
        'a bad photo of the {}.',
        'a {} in a video game.',
        'a origami {}.',
        'a photo of the small {}.',
        'art of the {}.',
        'a photo of the large {}.',
        'itap of a {}.',
    ],
    'ImageNet': [
        'a bad photo of the {}.',
        'a {} in a video game.',
        'a origami {}.',
        'a photo of the small {}.',
        'art of the {}.',
        'a photo of the large {}.',
        'itap of a {}.',
    ],
    'CIFAR10_local': [
        'a photo of a {}.',
        'a blurry photo of a {}.',
        'a black and white photo of a {}.',
        'a low contrast photo of a {}.',
        'a high contrast photo of a {}.',
        'a bad photo of a {}.',
        'a good photo of a {}.',
        'a photo of a small {}.',
        'a photo of a big {}.',
        'a photo of the {}.',
        'a blurry photo of the {}.',
        'a black and white photo of the {}.',
        'a low contrast photo of the {}.',
        'a high contrast photo of the {}.',
        'a bad photo of the {}.',
        'a good photo of the {}.',
        'a photo of the small {}.',
        'a photo of the big {}.',
    ],
    'CIFAR100_local': [
        'a photo of a {}',
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ],
    'Caltech101': [
        'a bad photo of the {}.',
        'a {} in a video game.',
        'a origami {}.',
        'a photo of the small {}.',
        'art of the {}.',
        'a photo of the large {}.',
        'itap of a {}.',
    ]

}

def load_clip_to_cpu(arch):
    backbone_name = arch
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

def gen_labels_with_descrptions(classes, descriptions):
    desc_ = []
    labels = []
    # classes = descriptions.keys() # uncomment this for sun397
    for i, classname in enumerate(classes):
        for desc in descriptions[classname]:
            desc_.append(desc)
            labels.append(i)
    return desc_, labels

def process_json(path_to_file, dataset):
    with open(path_to_file) as f:
        gpt3_prompts = json.load(f)
    classes = gpt3_prompts.keys()
    desc, labels_for_descriptions = gen_labels_with_descrptions(classes, descriptions=gpt3_prompts)

    dataset_templates = ds_specific_templates[dataset]
    templates, labels_for_templates = gen_labels_with_templates(classes, descriptions=dataset_templates)

    desc += templates
    labels_for_descriptions += labels_for_templates

    return classes, desc, labels_for_descriptions


def init_classifier_weights(classnames, dataset):
    clip_model = load_clip_to_cpu("ViT-B/32")
    clip_model.float()
    text_pr = "a satellite photo of {}"

    model = ALP(model=clip_model, classes=classnames, 
                           templates=[text_pr], ds_templates=ds_specific_templates[dataset], 
                           dataset_name=dataset, txt_cls='lafter')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
                    
    self.model.visual.classifier = nn.Parameter(torch.stack(zeroshot_weights, dim=1).to(args.device))        
    # delete unused modules
    del self.model.transformer, self.model.token_embedding, self.model.positional_embedding, self.model.ln_final, self.model.text_projection, self.model.logit_scale
    return


if __name__ == '__main__':
    dataset_json = '/l/users/sanoojan.baliah/Felix/RS_zero_shot/descriptions/generic/AID.json'
    dataset = os.path.splitext(dataset_json.split('/')[-1])[0]
    classnames, desc, labels_for_desc = process_json(dataset_json, dataset)
    init_classifier_weights(classnames, dataset)
    print("name")