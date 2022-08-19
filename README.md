# HCTransformers


PyTorch implementation for **"Attribute Surrogates Learning and Spectral Tokens Pooling in Transformers for Few-shot Learning"**.  
[[`arxiv`](https://arxiv.org/abs/2203.09064v1)]


## Prerequisites
This codebase has been developed with Python version 3.8, [PyTorch](https://pytorch.org/) version 1.9.0, CUDA 11.1 and torchvision 0.10.0. It has been tested on Ubuntu 20.04. 


## Datasets
coral dataset, split them into train, test, and val three folders. 


## Training
We provide the training code for 𝒎𝒊𝒏𝒊ImageNet, 𝒕𝒊𝒆𝒓𝒆𝒅ImageNet and CIFAR-FS, extending the **DINO** repo ([link](https://github.com/facebookresearch/dino)). 


### 1 Pre-train the First Transformer
To pre-train the first Transformer with attribute surrogates learning on 𝒎𝒊𝒏𝒊ImageNet from scratch with multiple GPU, run:
```
python -m torch.distributed.launch --nproc_per_node=8 main_hct_first.py --arch vit_small --data_path /path/to/coral_data/train --output_dir /path/to/HCTransformers/checkpoints_first/
```

### 2 Train the Hierarchically Cascaded Transformers
To train the Hierarchically Cascaded Transformers with sprectral token pooling on 𝒎𝒊𝒏𝒊ImageNet, run:
```
python -m torch.distributed.launch --nproc_per_node=8 main_hct_pooling.py --arch vit_small --data_path /path/to/mini_imagenet/train --output_dir /path/to/saving_dir --pretrained_weights /path/to/pretrained_weights
```

## Evaluation
To evaluate the performance of the first Transformer on 𝒎𝒊𝒏𝒊ImageNet 5-way 1-shot task, run:
```
python eval_hct_first.py --arch vit_small --server mini --partition test --checkpoint_key student --ckp_path /path/to/checkpoint_mini/ --num_shots 1
```

To evaluate the performance of the Hierarchically Cascaded Transformers on 𝒎𝒊𝒏𝒊ImageNet 5-way 5-shot task, run:
```
python eval_hct_pooling.py --arch vit_small --server mini_pooling --partition val --checkpoint_key student --ckp_path /path/to/checkpoint_mini_pooling/  --pretrained_weights /path/to/pretrained_weights_of_first_satge --num_shots 5
```


