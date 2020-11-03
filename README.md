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
Please make sure the path of datasets in config file is right.  

For example, to train a **DSHNet** model with Faster R-CNN R50-FPN for trainset of VisDrone:
```train
# Single GPU
python tools/train.py configs/faster_rcnn/vd_faster_rcnn_r101_fpn_tail.py --work-dir checkpoints/vd_faster_rcnn_r101_fpn_tail
``` 
Multi-gpu training and test are also supported as mmdetection.

## Test
```test
# Single GPU
python tools/test_lvis.py \
 ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
 ```
 
For example (assume that you have downloaded the corresponding chechkpoints file or train a model by ypurself to proper path), to evaluate the trained **DSHNet** model with Faster R-CNN R50-FPN for valset of VisDrone:
```test
# Single GPU
python tools/test.py configs/faster_rcnn/vd_faster_rcnn_r101_fpn_tail.py checkpoints/vd_faster_rcnn_r101_fpn_tail/latest.pth --eval bbox
 ```
## Results and models
Please refer to our paper for complete results.
|methods|backbone|map|map50|map75|maps|mapm|mapl|ped.|people|bicycle|car|van|truck|tricycle|awn.|bus|motor|model|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|FRCNN+FPN+DSHNet|R50|24.6|44.4|24.1|17.5|33.8|36.1|22.5|16.5|10.1|52.8|32.6|22.1|17.5|8.8|39.5|23.7|todo|
|FRCNN+FPN+DSHNet|R101|24.4|44.3|23.8|17.2|33.6|34.8|21.7|16.0|10.1|52.2|31.6|22.7|17.1|9.5|38.6|24.0|todo|
