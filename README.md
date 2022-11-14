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

### Download

You can download the pretrained models and datasets on [GoogleDrive](https://drive.google.com/drive/folders/1Yy_GH6_bydYPU6_JJzFQwig4LTh86VI4?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1WVdNccqDMnJ5k5Q__Y2dsg?pwd=gtuw) (gtuw).

Currently, we only provide gamma-corrected RGB images of our RS-Haze dataset.

The final file path should be the same as the following:

```
┬─ save_models
│   ├─ indoor
│   │   ├─ dehazeformer-b.pth
│   │   └─ ... (model name)
│   └─ ... (exp name)
└─ data
    ├─ RESIDE-IN
    │   ├─ train
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │       └─ ... (corresponds to the former)
    │   └─ test
    │       └─ ...
    └─ ... (dataset name)
```

## Training and Evaluation

In order to align the folder structure of each dataset, we rearrange the original dataset.

### Train

You can modify the training settings for each experiment in the `configs` folder.
Then run the following script to train the model:

```sh
python train.py --model (model name) --dataset (dataset name) --exp (exp name)
```

For example, we train the DehazeFormer-B on the ITS:

```sh
python train.py --model dehazeformer-b --dataset RESIDE-IN --exp indoor
```

TensorBoard will record the loss and evaluation performance during training.

### Test

Run the following script to test the trained model:

```sh
python test.py --model (model name) --dataset (dataset name) --exp (exp name)
```

For example, we test the DehazeFormer-B on the SOTS indoor set:

```sh
python test.py --model dehazeformer-b --dataset RESIDE-IN --exp indoor
```

Main test scripts can be found in `run.sh`.

### Predict

Run the following script to fetch the results:

```sh
python predict.py --model (model name) --folder (folder name) --exp (exp name)
```

For example, we fetch the results of DehazeFormer-B on the SOTS indoor set:

```sh
python predict.py --model dehazeformer-b --folder RESIDE-IN/test/hazy --exp indoor
```

## Notes

1. Currently, this repository provides roughly organized code, please send me an email (syd@zju.edu.cn) if you find problems. 
2. We found that the test results (RESIDE-OUT, RESIDE-6K) using opencv installed with conda are lower than pypi since they call different JPEG image codecs.
3. Unfortunately, because the synthetic dataset is not realistic enough (especially the RESIDE outdoor), the trained models (baselines and DehazeFormers) may not work well on real hazy images.
4. We are so sorry that we have no plans to release the code for the synthetic dataset method and MS images shortly.


## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{song2022vision,
  title={Vision Transformers for Single Image Dehazing},
  author={Song, Yuda and He, Zhuqing and Qian, Hui and Du, Xin},
  journal={arXiv preprint arXiv:2204.03883},
  year={2022}
}
```
