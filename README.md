<h1 align="center"> 
  <img src="assets/RH_logo.png" width="40%" />
</h1>

Relative Human (RH) contains **multi-person in-the-wild** RGB images with rich human annotations, including:  
 - **Depth layers (DLs):** relative depth relationship/ordering between all people in the image.  
 - **Age group classfication:** adults, teenagers, kids, babies.  
 - Others: **Genders**, **Bounding box**, **2D pose**.  

<p float="center">
  <img src="assets/depth_layer.png" width="20%" />
  <img src="assets/RH_demos.png" width="46%" />
  <img src="assets/RH_skeletons.png" width="30%" />
</p>

RH is introduced in CVPR 2022 paper [Putting People in their Place: Monocular Regression of 3D People in Depth](https://arxiv.org/abs/2112.08274).

 **[[Project Page]](https://arthur151.github.io/BEV/BEV.html) [[Video]](https://youtu.be/Q62fj_6AxRI) [[BEV Code]](https://github.com/Arthur151/ROMP)**

## Download

[[Google drive]](https://drive.google.com/drive/folders/1S6nOFmljodRsB0TFx4aIzssQ0NJ3xXCm?usp=share_link)   
[[Baidu drive]](https://pan.baidu.com/s/12z2rNU-Sex-LvS7AaV_Mfg?pwd=r3vh)  

## Leaderboard

See [Leaderboard](https://paperswithcode.com/sota/3d-depth-estimation-on-relative-human).

## Why do we need RH?

<p float="center">
  <img src="assets/RH_table.png" width="48%" />
</p>

Existing 3D datasets are poor in diversity of age and multi-person scenories. In contrast, RH contains richer subjects with explicit age annotations in the wild. We hope that RH can promote relative research, such as monocular depth reasoning, baby / child pose estimation, and so on. 

## How to use it?

We provide a toolbox for [data loading, visualization](demo.py), and [evaluation](RH_evaluation/evaluation.py). 

To run the demo code, please download the data and set the dataset_dir in [demo code](demo.py).

To use it for training, please refer to [BEV](https://github.com/Arthur151/ROMP) for details.

## Re-implementation

To re-implement RH results (in Tab. 1 of BEV paper), please first download the predictions from [here](https://github.com/Arthur151/Relative_Human/releases/download/Predictions/all_results.zip), then 
```
cd Relative_Human/
# BEV / ROMP / CRMH : set the path of downloaded results (.npz) in RH_evaluation/evaluation.py, then run
python -m RH_evaluation.evaluation

cd RH_evaluation/
# 3DMPPE: set the paths in eval_3DMPPE_RH_results.py and then run
python eval_3DMPPE_RH_results.py
# SMAP: set the paths in eval_SMAP_RH_results.py and then run
python eval_SMAP_RH_results.py
```

## Citation
Please cite our paper if you use RH in your research. 
```bibtex
@InProceedings{sun2022BEV,
author = {Sun, Yu and Liu, Wu and Bao, Qian and Fu, Yili and Mei, Tao and Black, Michael J},
title = {Putting People in their Place: Monocular Regression of {3D} People in Depth}, 
booktitle = {IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)}, 
year = {2022}
}
```
