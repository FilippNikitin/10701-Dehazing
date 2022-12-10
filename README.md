# Adverse Weather Image Processing (CMU 10701 Project)

> **Abstract:** 
Haze is a common phenomenon that affects the visibility and clarity of natural images. Aerosols
and adverse weather conditions such as dust, mist, fog, and snow can create haze. In normal day-to-
day life, this phenomenon can cause wrong object detection and serious traffic security issues. In
computer vision, a hazy image impacts the quality of analysis and thus results in unreliable high-level
computer vision applications such as self-driving cars. This is why image de-hazing is considered of
great importance in computer vision. In this project, we are focusing on using several architectures for image dehazing such as - the swin transformer based DehazeFormer, the multiple color channel based feature map from TheiaNet, and the transformer based sematic segmentation model HRNet.


## Workload
**Project initialization**:
- [x] Reproduce [DehazeFormer](https://github.com/IDKiro/DehazeFormer) results

**Experimental pipeline**:
- [x] Add convinient experiment logging with [WandB](https://wandb.ai/)
- [x] Add flexible experiment infrastructure based on [Pytorch Lightning](https://www.pytorchlightning.ai/)
- [x] Add flexible configuration file for experiments
- [x] Log number of parameters and computational complexity

**Features**:
- [x] Arbitrary number of loss functions
- [x] Arbitrary number of metrics
- [x] Quadruple color space

**Architectures**:
- [x] DehazeFormer
- [x] HRNet
- [x] HRTransformer
- [x] A-HRNet
- [x] TheiaNet
- [x] ESDNet

## Installation and usage

### Installation
```bash
conda create -n dehazing python=3.7
conda activate dehazing
pip install -r requirements.txt
```

### Usage
We provided various configuration's files for our porgect in the folder `configs`. Feel free to pick one of theme or 
create your own config file by following `configs/config.yaml`. 

To train the model you should run the next line:
```bash
python lightning.py configs/config.yaml
```
To test the model you should run this line:
```bash
python lightning.py configs/theianet.yaml --test --ckpt_path="your_checkpoint.ckpt"
```

## Experimental results

### Dataset

We are using one of the most common benchmarking dataset in image dehazing - 
the [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2?authuser=0) dataset. The RESIDE dataset
has three versions — RESIDE-V 0, RESIDE-standard, and RESIDE-β. IAs we want to compare our modified architecture and 
loss functions with the base architectures, we will be using the RESIDE-6k dataset, which used a combination of 3000 
indoor image pairs (ITS) and 3000 outdoor image pairs (OTS) for training and 1000 images from synthetic outdoor images 
(SOTS) for testing.

## Architectures



### Network Architecture
- The base architecture of DehazeFormer.
![DehazeFormer](figs/arch.png)
- Transformer based semantic segmentation arhitecture - HRNet.
![HRNet](figs/seg-hrnet.png)




## Notes
Currently, the code of this repository is under development. We will be updating architectures as we implement 
them as part of our project.


