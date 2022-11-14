# Adverse Weather Image Processing (CMU 10701 Project)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2204.03883) [![Dataset](https://img.shields.io/badge/GoogleDrive-Dataset-blue)](https://drive.google.com/drive/folders/1oaQSpdYHxEv-nMOB7yCLKfw2NDCJVtrx?usp=sharing) 
[![Model](https://img.shields.io/badge/GoogleDrive-Weight-blue)](https://drive.google.com/drive/folders/1gnQiI_7Dvy-ZdQUVYXt7pW0EFQkpK39B?usp=sharing)
[![BaiduPan](https://img.shields.io/badge/BaiduPan-Backup-orange)](https://pan.baidu.com/s/1WVdNccqDMnJ5k5Q__Y2dsg?pwd=gtuw)

> **Abstract:** 
Haze is a common phenomenon that affects the visibility and clarity of natural images. Aerosols
and adverse weather conditions such as dust, mist, fog, and snow can create haze. In normal day-to-
day life, this phenomenon can cause wrong object detection and serious traffic security issues. In
computer vision, a hazy image impacts the quality of analysis and thus results in unreliable high-level
computer vision applications such as self-driving cars. This is why image de-hazing is considered of
great importance in computer vision. In this project, we are focusing on using several architectures for image dehazing such as - the swin transformer based DehazeFormer, the multiple color channel based feature map from TheiaNet, and the transformer based sematic segmentation model HRNet.

### Network Architecture
- The base architecture of DehazeFormer.
![DehazeFormer](figs/arch.png)
- Transformer based semantic segmentation arhitecture - HRNet.
![HRNet](figs/seg-hrnet.png)

## Getting started

### Install

We test the code on PyTorch 1.10.2 + CUDA 11.3 + cuDNN 8.2.0.

1. Create a new conda environment
```
conda create -n pt1102 python=3.7
conda activate pt1102
```

2. Install dependencies
```
conda install pytorch=1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Dataset

We are using one of the most common benchmarking dataset in image dehazing - the [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2?authuser=0) dataset. The RESIDE dataset has three versions — RESIDE-V 0, RESIDE-standard, and RESIDE-β. IAs we want to compare our modified architecture and loss functions with the base architectures, we will be using the RESIDE-6k dataset, which used a combination of 3000 indoor image pairs (ITS) and 3000 outdoor image pairs (OTS) for training and 1000 images from synthetic outdoor images (SOTS) for testing. 

## Training and Evaluation

### Train

```sh
python train.py --model (model name) --dataset (dataset name) --exp (exp name)
```

For example, we train the model as - 

```sh
python train.py --model dehazeformer-b --dataset RESIDE-IN --exp indoor
```

We have used WanDB for recording the loss and evaluation performance (PSNR, SSIM) during training.

### Test

```sh
python test.py --model (model name) --dataset (dataset name) --exp (exp name)
```

For example,

```sh
python test.py --model dehazeformer-b --dataset RESIDE-IN --exp indoor
```

Main test scripts can be found in `run.sh`.


## Notes

- Currently, the code of this repository is under development. We will be updating architectures as we implement them as part of our project.


