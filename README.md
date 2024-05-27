# **NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces by Enforcing Zero Gaussian Curvature**

### [Project](https://qiujiedong.github.io/publications/NeurCADRecon/) | [Paper](https://arxiv.org/pdf/2404.13420.pdf)

**This repository is the official PyTorch implementation of our paper,  *NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces by Enforcing Zero Gaussian Curvature*.**

**This code is based on the SIREN, we also provide the implementation based on the IGR: [NeurCADRecon-IGR](https://github.com/QiujieDong/NeurCADRecon_IGR)**

<img src='./assets/teaser.png'>

## News
- :fire: This paper was accepted by [ACM TOG (SIGGRAPH 2024)](https://arxiv.org/abs/2404.13420)
- :star: July 29, 2024 (GMT -7): Gave a talk at [SIGGRAPH 2024](https://s2024.siggraph.org/) on NeurCADRecon.

## Requirements

- python 3.7
- CUDA 12.2
- pytorch 1.13.0

## Installation

```
git clone https://github.com/QiujieDong/NeurCADRecon.git
cd NeurCADRecon
```

## Preprocessing
Sampling and normalizing to [-0.5, 0.5]

```
cd pre_processing
python pre_data.py
```

- gt_path: The ground truth mesh of the CAD model.
- input_path: The input point cloud that need to be reconstructed.


## Overfitting

```angular2html
cd surface_reconstruction
python train_surface_reconstruction.py
```
All parameters are set in the ```surface_recon_args.py```.


## Cite

If you find our work useful for your research, please consider citing the following papers :)

```bibtex
@article{Dong2024NeurCADRecon,
author={Dong, Qiujie and Xu, Rui and Wang, Pengfei and Chen, Shuangmin and Xin, Shiqing and Jia, Xiaohong and Wang, Wenping and Tu, Changhe},
title={NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces by Enforcing Zero Gaussian Curvature},
journal={ACM Transactions on Graphics},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
year={2024},
month={July},
volume = {43},
number={4},
doi={10.1145/3658171},
keywords = {CAD model, unoriented point cloud, surface reconstruction, signed distance function, Gaussian curvature}
}
```


## Acknowledgments
Our code is inspired by [Neural-Singular-Hessian](https://github.com/bearprin/Neural-Singular-Hessian),  [SIREN](https://github.com/vsitzmann/siren), and [IGR](https://github.com/amosgropp/IGR).

