# LLVIP-Pose -- Leveraging Thermal Imaging for Robust Human Pose Estimation in Low-Light Vision

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
This is the official PyTorch implementation of the paper *"[Leveraging Thermal Imaging for Robust Human Pose Estimation in Low-Light Vision](https://sites.google.com/view/awss2024/accueil)"* (ACCVW 2024).


Human Pose Estimation (HPE) is becoming increasingly ubiquitous, finding applications in diverse fields such as surveillance and worker safety, healthcare, sport and entertainment. Despite substantial research in HPE within the visible domain, there is limited focus on thermal imaging for HPE, primarily due to the scarcity and annotation difficulty of thermal data. Thermal imaging offers significant advantages, including better performance in low-light conditions and enhanced privacy, which can lead to greater acceptance of monitoring systems. In this work, we introduce LLVIP-Pose, an extension of the existing LLVIP dataset, to include 2D single-image pose estimation for aligned night-time RGB and thermal images, containing approximately 26k annotated skeletons. We detail our annotation process and propose a novel metric for identifying and correcting poorly annotated skeletons. Furthermore, we present a comprehensive benchmark of top-down, bottom-up, and single-stage pose estimation models evaluated on both RGB and thermal images. Our evaluations demonstrate how pre-training on grayscale COCO data with data augmentation can benefit thermal pose estimation. The LLVIP-Pose dataset addresses the lack of thermal HPE datasets, providing a valuable resource for future research in this area.

## Installation:

Clone the repository and run the following command to install PyTorch, MMPose and MMDet within a conda enviroment.

```
source tools/install_conda.sh
```

## Data Preparation

Download the LLVIP images from the *"[LLVIP Website](https://bupt-ai-cz.github.io/LLVIP/)"*
Downdload the LLVIP-Pose from the *"[LLVIP-Pose Release](https://github.com/MickaelCormier/llvip-pose/releases/)"*

Once dowloaded, please organise the data in the data/ directory as follows.

```
|-data
    |- llvip
        |- annotations
            |- train_llvip.json
            |- test_llvip.json
            |- test_bbox_thermal_nms_50_llvip.json
            |- test_bbox_rgb_nms_50_llvip.json
            |- finetuning
                |- test_llvip_finetuning_13_kpts.json

        |- infrared
            |- 010001.jpg
            |- .......jpg
            |- .......jpg
            |- 260536.jpg
        |- visible
            |- 010001.jpg
            |- .......jpg
            |- .......jpg
            |- 260536.jpg
```

## Training

For training:

```
python tools/train.py <config-file> --amp
```

## Testing 

For testing:

```
python tools/test.py <config-file> <checkpoint-file> --work-dir <working_directory>
```


## Citation 
If you are using our dataset for your research, please cite our paper.
```
@InProceedings{Cormier_2024_ACCV,
    author    = {Cormier, Mickael and Ng Zhi Yi, Caleb and Specker, Andreas and Bla{\ss}, Benjamin and Heizmann, Michael and Beyerer, J{\"u}rgen},
    title     = {Leveraging Thermal Imaging for Robust Human Pose Estimation in Low-Light Vision},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV) Workshops},
    month     = {December},
    year      = {2024}
}
```
Since LLVIP-Pose Dataset uses the images from LLVIP Dataset, please cite their work as well.
```
@inproceedings{jia2021llvip,
title={LLVIP: A visible-infrared paired dataset for low-light vision},
author={Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
pages={3496--3504},
year={2021}
}
```

## Acknowledgment
This project is developed based on [mmpose](https://github.com/open-mmlab/mmpose).