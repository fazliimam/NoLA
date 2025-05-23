# NoLA
In the era of foundation models, CLIP excels at aligning text and visual modalities but often yields subpar visual features for fine-grained tasks. Conversely, SSL-pretrained models like DINO effectively extract rich visual features but require an additional supervised linear probing step, which depends on costly labeled data. This paper introduces NoLA (No Labels Attached), a label-free prompt-tuning method that enhances CLIP-based image classification using unlabelled images by leveraging DINO's visual features and large language models (LLMs).

Our approach involves three key steps: (i) Generating robust textual embeddings from LLMs to better represent object classes, facilitating effective zero-shot classification compared to CLIP's default prompts. (ii) Using these textual embeddings to create pseudo-labels for training an alignment module that integrates LLM embeddings with DINO's visual features. (iii) Prompt-tuning CLIP's vision encoder through DINO-assisted supervision via the trained alignment module.

This framework effectively combines the strengths of visual and textual foundation models, achieving an average gain of 3.6% over the state-of-the-art LaFter across 11 diverse image classification datasets.

## Installation

Our code has been tested in an environment with `python 3.8.8` and `pytorch 2.3.1` compiled with `CUDA 12.1`. 

As a first step, install `dassl` library (under `NoLA/`) in your environment by following the instructions [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation).

To further install all other dependencies, please run the following command, after having your environment activated:

```
pip install -r requirements.txt
```

## Datasets

Under `NoLA/` first make an empty data folder: 

```
mkdir data
```

Then download and structure your datasets according to the instructions provided in the [CoOp](https://github.dev/KaiyangZhou/CoOp)
official repository. All the `12` datasets should be present in the `data/` directory.

## Descriptions

The class-wise descriptions for the `12` datasets are present in `descriptions/generic` directory. 
The code for generating these descriptions is also provided in the `descriptions/generate_descriptions.py` file.

## Experiments

### NoLA
To run the full `NoLA` pipeline, please run the following command:

```
bash scripts/nola_train.sh <dataset_name>
```

where `<dataset_name>` can be `dtd`, `eurosat`, etc.

### Trained Checkpoints
Trained checkpoints for all the datasets can be found [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/mohamed_imam_mbzuai_ac_ae/EkGSn1Ato_dGj9DFDz2cIjMBbzngkBhr9Cq03A9MKw-C3A?e=ceScao).

## Acknowledgements 
Our code is based on [LaFTer](https://github.com/jmiemirza/LaFTer) and [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). We thank the authors for releasing their code.


