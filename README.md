# Towards Resolving the Challenge of Long-tail Distribution in UAV Images for Object Detection

## Introduction
This repo is the implementation for WACV 2021 paper: **Towards Resolving the Challenge of Long-tail Distribution in UAV Images for Object Detection**.
![Framework](fig2.png)
## Requirements
### 1. Environment:
The requirements are exactly the same as [mmdetection v2.3.0rc0+8194b16](https://github.com/open-mmlab/mmdetection/tree/v2.3.0). We tested on on the following settings:

- python 3.7.7
- cuda 10.1
- pytorch 1.5.0 
- torchvision 0.6.0
- mmcv 1.0.4

With settings above, please refer to [offical guide of mmdetection](https://github.com/open-mmlab/mmdetection/blob/v2.3.0/docs/install.md) for installation.
### 2. Data:
Please download trainset and valset of [VisDrone2020-DET dataset](http://aiskyeye.com/download/object-detection/) and [UAVDT-Benchmark-M](https://sites.google.com/site/daviddo0323/projects/uavdt), then unzip all the files and put them under proper paths.

In order to make better use of mmdetection, please convert the datasets to coco format.

## Training

Both training and test commands are exactly the same as mmdetection.
```train
# Single GPU
python tools/train.py ${CONFIG_FILE}
```
Please make sure the path of datasets of config .py file is right.  

For example, to train a **DSHNet** model with Faster R-CNN R50-FPN for trainset of VisDrone:
```train
# Single GPU
python tools/train.py configs/faster_rcnn/vd_faster_rcnn_r101_fpn_tail.py --work-dir checkpoints/vd_faster_rcnn_r101_fpn_tail
``` 
Multi-gpu training and test are also supported as mmdetection.

## Testing
```test
# single gpu test
python tools/test_lvis.py \
 ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
 ```
 
For example (assume that you have downloaded the corresponding chechkpoints file or train a model by ypurself to proper path), to evaluate the trained **DSHNet** model with Faster R-CNN R50-FPN for valset of VisDrone:
```eval
# single-gpu testing
python tools/test.py configs/faster_rcnn/vd_faster_rcnn_r101_fpn_tail.py checkpoints/vd_faster_rcnn_r101_fpn_tail/latest.pth --eval bbox
 ```
