# 3D Multi-Object-Tracking group 42

<b>3D Multi-Object Tracking: for DLAV CIVIL-459 by Johan Lagerby and Axel Englund</b>
Built upon [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT) and using [DINOv2](https://github.com/facebookresearch/dinov2) for extracting visual features for tracking.
AB3DMOT, the method we built our contribution into can be found here "[3D Multi-Object Tracking: A Baseline and New Evaluation Metrics](http://www.xinshuoweng.com/papers/AB3DMOT/proceeding.pdf)".

<img src="dinov2-mot.gif" alt="GIF Example" width="800" height="300">

## Contribution Overview
Our contribution is to add visual appearance re-identification to an existing mulitple 3D tracking algorithm that doesn't use visual features. We choose AB3DMOT ([paper](http://www.xinshuoweng.com/papers/AB3DMOT/proceeding.pdf)) as our baseline model from which we added a visual appearance functionality. To get the visual features from a detection, we use Facebook Research's visual deep model DINOv2 ([paper](https://arxiv.org/abs/2304.07193)). We calculate the similarity between detections and tracklets using the cosine similarity and uses that as cost/affinity. We then add a hyper-parameter $\alpha$ which controls the how much of our contribution to use in calculation of the cost matrix: 
$$\text{Total cost} = (1-\alpha) \cdot \text{AB3DMOT cost} + \alpha \cdot \text{Our contribution cost}$$

Our contribution doesn't need the 3D bounding boxes of the detections to generate the embeddings. But rather we use the 2D bounding boxes, to crop the frame into smaller images - containing only the detection - which are used to pass through the visual deep model.

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

NOTE: This "Install" is largely based on the [AB3DMOT install](https://github.com/xinshuoweng/AB3DMOT/blob/master/docs/INSTALL.md).

## Dataset
We used the [KITTI](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) dataset. Mainly you need left color images, velodyne point cloud data, GPS/IMU data, training labels, and camera calibration data. Furthermore, it is important that the dataset lies in the correct format in the repo:
```
group42_3DMOT
├── data
│   ├── KITTI
│   │   │── tracking
│   │   |   │── training
│   │   │   │   ├──calib & velodyne & label_02 & image_02 & oxts
│   │   │   │── testing
│   │   │   │   ├──calib & velodyne & image_02 & oxts
├── AB3DMOT_libs
├── configs
```
If one wants to run this on SCITAS, soft symbolic links already exists here that takes you to the correct datasets already uploaded to SCITAS.

This code uses the KITTI 3D Object Detection Challenge format for the detections, but with some switch in order:

Frame |   Type  |   2D BBOX (x1, y1, x2, y2)  | Score |    3D BBOX (h, w, l, x, y, z, rot_y)      | Alpha | 
------|:-------:|:---------------------------:|:-----:|:-----------------------------------------:|:-----:|
 0    | 2 (car) | 726.4, 173.69, 917.5, 315.1 | 13.85 | 1.56, 1.58, 3.48, 2.57, 1.57, 9.72, -1.56 | -1.82 | 
 
 More info found in the object development toolkit here: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
 
 NOTE: This dataset setup is from [AB3DMOT KITTI dataset setup](https://github.com/xinshuoweng/AB3DMOT/blob/master/docs/KITTI.md)
## 3D Multi-Object Tracking

For the embeddings (feature vectors) for the detections, we choose to generate these before we actually run the tracking. We opt for this approach to save time and avoid generating embeddings with each run. We submit jobs to the SCITAS cluster for this purpose and store the embeddings in .txt files (in either the "embeddings_val" or "embeddings_test_split" folder). As a result, this implementation functions as a batch process rather than real-time, although it could be run online if executed on a sufficiently powerful computer. However, the code would require some modifications in that case.

# Data preprocessing (embedding generation)

To save the embeddings from the KITTI dataset with the provided Point RCNN detections, run the following code. 
WARNING: This part was run using cuda on the SCITAS cluster. Note that this requires alot of computing power.
~~~shell
python3 main.py --dataset KITTI --det_name pointrcnn --get_embeddings 
~~~
or 
~~~shell
python3 main.py --dataset KITTI --det_name pointrcnn --get_embeddings --split test
~~~
to run the test split.

NOTE: that the embeddings for the KITTI MOT validation set with the provided PointRCNN detections and the test set with the provided PointRCNN detections have already been generated and can be found under "embeddings_val" and "embeddings_test_split" respectively. 

# Inference

To run our tracker we follow the almost the same instructions as given by the author for AB3DMOT.

NOTE: The evaluation of the results will be performed with a weighting $\alpha$ that is the ratio between using the metric from AB3DMOT and DINOv2. This ratio is between 0 and 1 depending on which part is contributing the most, 0 means the default AB3DMOT metric is used and 1 means only the DINOv2 metric is used.

"To run our tracker on the KITTI MOT validation set with the provided PointRCNN detections." here with $\alpha = 0.5$
~~~shell
python3 main.py --dataset KITTI --det_name pointrcnn --alpha 0.5
~~~
"In detail, running above command will generate a folder named "pointrcnn_val_H1" that includes results combined from all categories, and also folders named "pointrcnn_category_val_H1" representing results for each category. Under each result folder, "./data_0" subfolders are used for MOT evaluation, which follow the format of the KITTI Multi-Object Tracking Challenge (format definition can be found in the tracking development toolkit here: http://www.cvlibs.net/datasets/kitti/eval_tracking.php). Also, "./trk_withid_0" subfolders are used for visualization only, which follow the format of KITTI 3D Object Detection challenge except that we add an ID in the last column."
### Results from experiment
Before we decided to use DINOv2 for our visual feature extraction. We examined the power of the feature representation of detections. We took 3 full-body pictures of two humans named Axel and Johan from different angles (these can be found under "experiments_im/), than ran these through two visual models: DINOv2 ([github](https://github.com/facebookresearch/dinov2), [paper](https://arxiv.org/abs/2304.07193)) and Segment-Anything ([github](https://github.com/facebookresearch/segment-anything), [paper](https://ai.facebook.com/research/publications/segment-anything/)). Both are new (of this date 31/05-2023) deep learning models developed by Facebook-research. Segment-Anything is designed to be promptable and can transfer zero-shot to new image distributions and tasks. The DINOv2 paper however, explores the idea of all-purpose visual features in computer vision and proposes a self-supervised approach for pretraining on a large curated image dataset using a ViT (Vision Transformers) model. So at first glance the two seems to be good candidates for our task. After running our test images thorugh the models, we used two different metrics to evaluate the similiarty between the images - cosine similarity and the euclidean distance. Optimum would be good similarity between images depciting the same person and bad similarty scores between images that depict different persons.

To get the same results run experiments.py - we ran this script locally and never tried running it on SCITAS. Furthermore, you need to [download](https://github.com/facebookresearch/segment-anything#model-checkpoints) weights for the Segment-anything model (we used [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)) and:

~~~shell
pip install git+https://github.com/facebookresearch/segment-anything.git
~~~
Here is the results from this experiments:

#### DINOV2 Cosine-Similarity (-1 to 1) 

|                                  |axel1|axel2|axel3|johan1|johan2|johan3|
|----------------------------------|:----:|:----:|:----:|:----:|:----:|:----:|
| axel1                            | 1.00| 0.69| 0.88| 0.44| 0.58| 0.67|
| axel2                            | 0.69| 1.00| 0.63| 0.45| 0.74| 0.48|
| axel3                            | 0.88| 0.63| 1.00| 0.53| 0.57| 0.75|
| johan1                           | 0.44| 0.45| 0.53| 1.00| 0.62| 0.52|
| johan2                           | 0.58| 0.74| 0.57| 0.62| 1.00| 0.71|
| johan3                           | 0.67| 0.48| 0.75| 0.52| 0.71| 1.00|

#### DINOv2 Euclidean-Similarity

|                      |axel1|axel2|axel3|johan1|johan2|johan3|
|----------------------|:----:|:----:|:----:|:----:|:----:|:----:|
| axel1                | 0.00|35.84|22.53|48.24|41.87|36.87|
| axel2                |35.84| 0.00|39.57|47.87|32.98|46.45|
| axel3                |22.53|39.57| 0.00|44.40|42.40|32.29|
| johan1               |48.24|47.87|44.40| 0.00|39.80|44.77|
| johan2               |41.87|32.98|42.40|39.80| 0.00|34.97|
| johan3               |36.87|46.45|32.29|44.77|34.97| 0.00|

#### SEGMENT-ANYHTING-MODEL Cosine-Similarity (-1 to 1)

|        |axel1|axel2|axel3|johan1|johan2|johan3|
|--------|:---:|:---:|:---:|:----:|:----:|:----:|
| axel1  | 1.00| 0.64| 0.79| 0.74 | 0.69 | 0.76 |
| axel2  | 0.64| 1.00| 0.63| 0.72 | 0.81 | 0.59 |
| axel3  | 0.79| 0.63| 1.00| 0.73 | 0.69 | 0.75 |
| johan1 | 0.74| 0.72| 0.73| 1.00 | 0.78 | 0.68 |
| johan2 | 0.69| 0.81| 0.69| 0.78 | 1.00 | 0.65 |
| johan3 | 0.76| 0.59| 0.75| 0.68 | 0.65 | 1.00 |

#### SEGMENT-ANYHTING-MODEL Euclidean-Similarity

|        |axel1|axel2|axel3|johan1|johan2|johan3|
|--------|:---:|:---:|:---:|:----:|:----:|:----:|
| axel1  | 0.00|98.73|74.74|84.41 |91.50 |81.33 |
| axel2  |98.73| 0.00|100.17|86.96 |71.35 |105.02|
| axel3  |74.74|100.17| 0.00|86.08 |91.61 |82.40 |
| johan1 |84.41|86.96|86.08| 0.00 |76.58 |92.15 |
| johan2 |91.50|71.35|91.61|76.58 | 0.00 |96.73 |
| johan3 |81.33|105.02|82.40|92.15 |96.73 | 0.00 |

As seen in the tables, DINOv2 generated better embeddings since the cosine similarities were, in general, closer to each other for each respective image. Additionally, performing a forward pass with "Segment-Anything" takes considerably longer than with DINOv2. Hence, we choose DINOv2 for this implementation.
### Evaluation
To get the evalutation metrics we run:

~~~shell
python3 scripts/KITTI/evaluate.py pointrcnn_val_H1 1 3D 0.25
~~~
Which runs the evaluation on the KITTI MOT validation set with a threshold of 0.25 3D IoU.

NOTE: To try different $\alpha$ the user must rerun

~~~shell
python3 main.py --dataset KITTI --det_name pointrcnn --alpha X
~~~

before running the evalutation script.

#### PointRCNN + AB3DMOT (KITTI val set)

Results evaluated with the 0.25 3D IoU threshold (BASELINE):

Category       | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP  |  FN  |  FPS 
--------------- |:------:|:------:|:------:|:---:|:----:|:----:|:----:|:----:|
 *Car*          | 93.34  | 86.47  |  79.40 |  0  | 15   | 368  | 766  | 108.7
 *Pedestrian*   | 82.73  | 73.86  |  67.58 |  4  | 62   | 589  | 1965 | 119.2
 *Cyclist*      | 93.78  | 84.79  |  77.23 |  1  | 3    | 114  | 90   | 980.7
 *Overall*      | 89.62  | 81.71  |  74.74 |  5  | 80   | 1071 | 2821 | -

#### PointRCNN + AB3DMOT + Dinov2 (KITTI val set), $\alpha = 0.25$

Results evaluated with the 0.25 3D IoU threshold with $\alpha = 0.25$.

Category       | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP  |  FN  |  FPS 
--------------- |:------:|:------:|:------:|:---:|:----:|:----:|:----:|:----:|
 *Car*          | 93.14  | 86.25  |  79.30 |  0  | 19   | 385  | 767  | -
 *Pedestrian*   | 69.41  | 63.65  |  66.66 |  6  | 163  | 585  | 2967 | -
 *Cyclist*      | 46.26  | 39.09  |  78.50 |  31 | 54   | 74   | 716  | -
 *Overall*      | 69.60  | 62.99  |  74.82 |  37 | 236  | 1044 | 5217 | -


#### PointRCNN + AB3DMOT + Dinov2 (KITTI val set), $\alpha = 0.5$

Results evaluated with the 0.25 3D IoU threshold with $\alpha = 0.5$.

Category       | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP  |  FN  |  FPS 
--------------- |:------:|:------:|:------:|:---:|:----:|:----:|:----:|:----:|
 *Car*          | 91.18  | 81.98  |  77.39 |  1  | 38   | 668  | 841  | -
 *Pedestrian*   | 50.16  | 39.97  |  65.04 |  14 | 227  | 1148 | 4713 | -
 *Cyclist*      | 28.63  | 22.85  |  79.02 |  15 | 31   | 38   | 987  | -
 *Overall*      | 67.51  | 48.27  |  73.82 |  30 | 296  | 1854 | 6541 | -

#### PointRCNN + AB3DMOT + Dinov2 (KITTI val set), $\alpha = 1$

Results evaluated with the 0.25 3D IoU threshold with $\alpha = 1$.

Category       | sAMOTA |  MOTA  |  MOTP  | IDS | FRAG |  FP  |  FN  |  FPS 
--------------- |:------:|:------:|:------:|:---:|:----:|:----:|:----:|:----:|
 *Car*          | 78.53  | 69.52  |  75.08 |  21 | 94   | 1237 | 1296 | -
 *Pedestrian*   | 0.57   | 0.57   |  53.98 |  9  | 39   | 172  | 9550 | -
 *Cyclist*      | 6.18   | 4.60   |  81.89 |  5  | 8    | 42   | 1239 | -
 *Overall*      | 28.43  | 24.89  |  70.32 |  35 | 141  | 1451 | 12085| -

## Finally
From the results we can say that our contribution greatly decreases many of the evaluation metrics. Most notably we see a decrease in the sAMOTA (scaled Average Multi Object Tracking Accuracy) accuracy metric, especially for the Pedestrian and Cyclist categories and this is implied by the large number of false positives produced. This indicates that when using more of our contribution, the tracker prediction is prone to actually miss the ground truth when it exists. Either our contribution is too harsh with the conditions for the predictions or more likely is the scaling of our metric when combining the metrics. When we implemented our contribution it was discovered that AB3DMOT uses two difference metrics, General Intersection over Union (GIoU) for the car and 3D distance (dist3d) for the pedestrian and cyclist. For the car, our metric was easily implemented due to that GIoU have the same scaling as our metric (-1 to 1), but for dist3d the scaling was, for us at least, arbitrary. It was initially thought to have a scale of 0-100, and thus we scaled our metric accordingly. But when reflecting over the result, this might have been a faulty scale. This is most likely the primary cause of the large difference. Interestingly enough, we actually improved the MOTP (Multi Object Tracking Precision) precision metric for the cyclist. This gives us a hint that the tracker is more consistent with its predictions but with worse accuracy. Thus, it would be wise to carefully evaluate the actual scaling of the 3D distance metric in order to improve our results with regards to the pedestrian and cyclist. 


We could probably improve the result if we fine-tuned the visual models on our dataset. This would be the natural next step. However, being new to projects of this magnitude, we initially found it difficult working on a project this size and to just implement the contributions we have made thus far. Another challenge has been working on SCITAS, both the author's personal computers run on windows, thus neither were epsecially comfortable in the beginning working on Linux machines, and sending jobs to the clusters seemed almost impossible. Also adding the KITTI dataset to SCITAS seemed like quite a hassle in the beginning.

However, we are feel that we have grown more comfortable and proficient with working on existing large-scale projects, working with remote connections on Linux systems, sending jobs to clusters, and managing substantial datasets. And if we would have done it again, we would probably start with fine-tuning a existing visual model. We would probably try a model with even stronger feature representations such as CLIP from OpenAI, and compare it with others. Additionally, we would try to make it truely online, and not pre-generate the embeddings before running the tracker.

Despite the difficulties, we are proud of the progress we have achieved and the insights we have gained.



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
