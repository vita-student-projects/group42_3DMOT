# 3D Multi-Object-Tracking group 42

<b>3D Multi-Object Tracking: for DLAV CIVIL-459 by Johan Lagerby and Axel Englund</b>
Built upon [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT) and using [DINOv2](https://github.com/facebookresearch/dinov2) for extracting visual features for tracking.

![GIF Example](our_viz.gif)

## System Requirements

This code has only been tested on the following combination of MAJOR pre-requisites. Please check beforehand. IMPORTANT!

* Ubuntu 18.04
* Python 3.7.7

## Install
Clone the code:
~~~shell
git clone https://github.com/vita-student-projects/group42_3DMOT.git
~~~
To install the required dependencies on the virtual environment of the python, please run the following command at the root of this code:

```
pip3 install venv
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```
Additionally, this code depends on the authors toolbox: https://github.com/xinshuoweng/Xinshuo_PyToolbox. Please install the toolbox by:

*1. Clone the github repository.*
~~~shell
git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox
~~~

Please add the path to the code to your PYTHONPATH in order to load the library appropriately. For example, if the code is located at /home/user/workspace/code/group42_3DMOT, please add the following to your ~/.profile:
```
export PYTHONPATH=${PYTHONPATH}:/home/user/workspace/code/group42_3DMOT
export PYTHONPATH=${PYTHONPATH}:/home/user/workspace/code/group42_3DMOT/Xinshuo_PyToolbox
```
## 3D Multi-Object Tracking
TODO: Explain how we forward the embeddings of the detections in the script and store them.

In model.py we define the function that gathers the embeddings from the detections. Etc...

To save the embeddings from the KITTI dataset with the provided Point RCNN detections,run the following code. 
WARNING: This part was run using cuda on the SCITAS cluster. Note that this requires alot of computing power.
~~~shell
python3 main.py --dataset KITTI --det_name pointrcnn --get_embeddings 
~~~
To run our tracker we follow the same instructions as given by the author for AB3DMOT.

NOTE: The evaluation of the results will be performed with a weighting alpha that is the ratio between using the metric from AB3DMOT and Dinov2. This ratio is between 0 and 1 depending on which part is contributing the most, 0 means the default AB3DMOT metric is used and 1 means only the Dinov2 metric is used.


"To run our tracker on the KITTI MOT validation set with the provided PointRCNN detections."
~~~shell
python3 main.py --dataset KITTI --det_name pointrcnn --alpha 0.5
~~~
"In detail, running above command will generate a folder named "pointrcnn_val_H1" that includes results combined from all categories, and also folders named "pointrcnn_category_val_H1" representing results for each category. Under each result folder, "./data_0" subfolders are used for MOT evaluation, which follow the format of the KITTI Multi-Object Tracking Challenge (format definition can be found in the tracking development toolkit here: http://www.cvlibs.net/datasets/kitti/eval_tracking.php). Also, "./trk_withid_0" subfolders are used for visualization only, which follow the format of KITTI 3D Object Detection challenge except that we add an ID in the last column."

#### PointRCNN + AB3DMOT (KITTI val set)

Results evaluated with the 0.25 3D IoU threshold:

Category       | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP  |  FN  |  FPS 
--------------- |:------:|:------:|:------:|:---:|:----:|:----:|:----:|:----:|
 *Car*          | 93.34  | 86.47  |  79.40 |  0  | 15   | 368  | 766  | 108.7
 *Pedestrian*   | 82.73  | 73.86  |  67.58 |  4  | 62   | 589  | 1965 | 119.2
 *Cyclist*      | 93.78  | 84.79  |  77.23 |  1  | 3    | 114  | 90   | 980.7
 *Overall*      | 89.62  | 81.71  |  74.74 |  5  | 80   | 1071 | 2821 | -

#### PointRCNN + AB3DMOT + Dinov2 (KITTI val set), alpha = 0.25

Results evaluated with the 0.25 3D IoU threshold with alpha = 0.25.

Category       | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP  |  FN  |  FPS 
--------------- |:------:|:------:|:------:|:---:|:----:|:----:|:----:|:----:|
 *Car*          | 93.14  | 86.25  |  79.30 |  0  | 19   | 385  | 767  | -
 *Pedestrian*   | 69.41  | 63.65  |  66.66 |  6  | 163  | 585  | 2967 | -
 *Cyclist*      | 46.26  | 39.09  |  78.50 |  31 | 54   | 74   | 716  | -
 *Overall*      | 69.60  | 62.99  |  74.82 |  37 | 236  | 1044 | 5217 | -


#### PointRCNN + AB3DMOT + Dinov2 (KITTI val set), alpha = 0.5

Results evaluated with the 0.25 3D IoU threshold with alpha = 0.5.

Category       | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP  |  FN  |  FPS 
--------------- |:------:|:------:|:------:|:---:|:----:|:----:|:----:|:----:|
 *Car*          | 91.18  | 81.98  |  77.39 |  1  | 38   | 668  | 841  | -
 *Pedestrian*   | 50.16  | 39.97  |  65.04 |  14 | 227  | 1148 | 4713 | -
 *Cyclist*      | 28.63  | 22.85  |  79.02 |  15 | 31   | 38   | 987  | -
 *Overall*      | 67.51  | 48.27  |  73.82 |  30 | 296  | 1854 | 6541 | -

#### PointRCNN + AB3DMOT + Dinov2 (KITTI val set), alpha = 1

Results evaluated with the 0.25 3D IoU threshold with alpha = 1.

Category       | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP  |  FN  |  FPS 
--------------- |:------:|:------:|:------:|:---:|:----:|:----:|:----:|:----:|
 *Car*          | 78.53  | 69.52  |  75.08 |  21 | 94   | 1237 | 1296 | -
 *Pedestrian*   | 0.57   | 0.57   |  53.98 |  9  | 39   | 172  | 9550 | -
 *Cyclist*      | 6.18   | 4.60   |  81.89 |  5  | 8    | 42   | 1239 | -
 *Overall*      | 28.43  | 24.89  |  70.32 |  35 | 141  | 1451 | 12085| -

<!-- # AB3DMOT

<!-- <b>3D Multi-Object Tracking: A Baseline and New Evaluation Metrics (IROS 2020, ECCVW 2020)</b>

This repository contains the official python implementation for our full paper at IROS 2020 "[3D Multi-Object Tracking: A Baseline and New Evaluation Metrics](http://www.xinshuoweng.com/papers/AB3DMOT/proceeding.pdf)" and short paper "[AB3DMOT: A Baseline for 3D Multi-Object Tracking and New Evaluation Metrics](http://www.xinshuoweng.com/papers/AB3DMOT_eccvw/camera_ready.pdf)" at ECCVW 2020. Our project website and video demos are [here](http://www.xinshuoweng.com/projects/AB3DMOT/). If you find our paper or code useful, please cite our papers:

```
@article{Weng2020_AB3DMOT, 
author = {Weng, Xinshuo and Wang, Jianren and Held, David and Kitani, Kris}, 
journal = {IROS}, 
title = {{3D Multi-Object Tracking: A Baseline and New Evaluation Metrics}}, 
year = {2020} 
}
```
```
@article{Weng2020_AB3DMOT_eccvw, 
author = {Weng, Xinshuo and Wang, Jianren and Held, David and Kitani, Kris}, 
journal = {ECCVW}, 
title = {{AB3DMOT: A Baseline for 3D Multi-Object Tracking and New Evaluation Metrics}}, 
year = {2020} 
}
```

<img align="center" width="100%" src="https://github.com/xinshuoweng/AB3DMOT/blob/master/main1.gif">
<img align="center" width="100%" src="https://github.com/xinshuoweng/AB3DMOT/blob/master/main2.gif">

## Overview

- [News](#news)
- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Demo on KITTI](#quick-demo-on-kitti)
- [Benchmarking](#benchmarking)
- [Acknowledgement](#acknowledgement)

## News

- Feb. 27, 2022: Added support to the nuScenes dataset and updated README
- Feb. 26, 2022: Refactored code libraries and signficantly improved performance on KITTI 3D MOT
- Aug. 06, 2020: Extend abstract (one oral) accepted at two ECCV workshops: [WiCV](https://sites.google.com/view/wicvworkshop-eccv2020/), [PAD](https://sites.google.com/view/pad2020/accepted-papers?authuser=0)
- Jul. 05, 2020: 2D MOT results on KITTI for all three categories released
- Jul. 04, 2020: Code modularized and a minor bug in KITTI evaluation for DontCare objects fixed
- Jun. 30, 2020: Paper accepted at IROS 2020
- Jan. 10, 2020: New metrics sAMOTA added and results updated
- Aug. 21, 2019: Python 3 supported
- Aug. 21, 2019: 3D MOT results on KITTI "Pedestrian" and "Cyclist" categories released
- Aug. 19, 2019: A minor bug in orientation correction fixed
- Jul. 9, 2019: Code and 3D MOT results on KITTI "Car" category released, support Python 2 only

## Introduction

3D multi-object tracking (MOT) is an essential component technology for many real-time applications such as autonomous driving or assistive robotics. However, recent works for 3D MOT tend to focus more on developing accurate systems giving less regard to computational cost and system complexity. In contrast, this work proposes a simple yet accurate real-time baseline 3D MOT system. We use an off-the-shelf 3D object detector to obtain oriented 3D bounding boxes from the LiDAR point cloud. Then, a combination of 3D Kalman filter and Hungarian algorithm is used for state estimation and data association. Although our baseline system is a straightforward combination of standard methods, we obtain the state-of-the-art results. To evaluate our baseline system, we propose a new 3D MOT extension to the official KITTI 2D MOT evaluation along with two new metrics. Our proposed baseline method for 3D MOT establishes new state-of-the-art performance on 3D MOT for KITTI, improving the 3D MOTA from 72.23 of prior art to 76.47. Surprisingly, by projecting our 3D tracking results to the 2D image plane and compare against published 2D MOT methods, our system places 2nd on the official KITTI leaderboard. Also, our proposed 3D MOT method runs at a rate of 214.7 FPS, 65 times faster than the state-of-the-art 2D MOT system. 

## Installation

Please follow carefully our provided [installation instructions](docs/INSTALL.md), to avoid errors when running the code.

## Quick Demo on KITTI

To quickly get a sense of our method's performance on the KITTI dataset, one can run the following command after installation of the code. This step does not require you to download any dataset (a small set of data is already included in this code repository).

```
python3 main.py --dataset KITTI --split val --det_name pointrcnn
python3 scripts/post_processing/trk_conf_threshold.py --dataset KITTI --result_sha pointrcnn_val_H1
python3 scripts/post_processing/visualization.py --result_sha pointrcnn_val_H1_thres --split val
```

## Benchmarking

We provide instructions (inference, evaluation and visualization) for reproducing our method's performance on various supported datasets ([KITTI](docs/KITTI.md), [nuScenes](docs/nuScenes.md)) for benchmarking purposes. 

### Acknowledgement

The idea of this method is inspired by "[SORT](https://github.com/abewley/sort)" -->