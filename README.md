# [prnet-tf2](https://github.com/heathentw/prnet-tf2)

PRNet (Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network, published in ECCV 2018) implemented in Tensorflow 2.0+. This is an unofficial implementation.

Original Paper: &nbsp; [Arxiv](https://arxiv.org/abs/1803.07835) &nbsp; [ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Feng_Joint_3D_Face_ECCV_2018_paper.pdf)

Offical Implementation: &nbsp; [PyTorch](https://github.com/YadiraF/PRNet)


<img src="Data/demo.gif">

****

## Contents

* [Installation](#Installation)
* [Training](#Training)
* [Testing](#Testing)
* [References](#References)


## Installation

Create a new python virtual environment by [Anaconda](https://www.anaconda.com/) or just use pip in your python environment and then clone this repository as following.

### Clone this repo
```bash
git clone git@https://github.com/heathentw/prnet-tf2.git
cd prnet-tf2
```

### Conda
```bash
conda env create -f environment.yml
conda activate prnet-tf2
```

### Pip

```bash
pip install -r requirements.txt
```


****

## Training

The training implementation use [300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) as training data, same as in the original paper. You can download the data-set and generate the position-maps with the [code](https://github.com/YadiraF/face3d/blob/master/examples/8_generate_posmap_300WLP.py) provided by the author, or download the generated [data-set](https://drive.google.com/open?id=1SCMfQ7Xs3BRgRxs62g_J1dMJkdNi0R52).

Arrange the data-set folders as follow:
```bash
./Data
├────── posmap
│       ├─ AFW_Flip
│       │  ├── AFW_1051618982_1_0.npy 
│       │  └── ...
│       ├─ AFW
│       ├─ HELEN_Flip
│       ├─ HELEN
│       ├─ IBUG_Flip
│       ├─ IBUG
│       ├─ LFPW_Flip
│       ├─ LFPW
└── ...
```



You can modify your own dataset path or other settings of model in [./configs/*.yaml](https://github.com/peteryuX/esrgan-tf2/tree/master/configs) for training and testing, which like below.

```python
# general setting
batch_size: 16
input_size: 256
num_workers: 0
ch_size: 3
sub_name: 'prnet'
pretrain_name: 

# dataset setting
train_dataset:
    name: '300WLP'
    path: './Data/posmap'
    num_samples: 122450

# training setting
epoch: 100

lr_G: !!float 1e-4
lr_steps: [100000, 200000, 300000, 400000]
lr_rate: 0.5

adam_beta1_G: 0.9
adam_beta2_G: 0.99

log_steps: 10
save_steps: 100

uv_weight_mask: './Data/uv-data/uv_weight_mask.png'
```

Note:
- The `sub_name` is the name of outputs directory used in checkpoints and logs folder. (make sure of setting it unique to other models)
- The `save_steps` is the number interval steps of saving checkpoint file.

#### Run training

```bash
python train.py --cfg_path ./configs/prnet.yaml --gpu 0
```


## Testing

For testing PRNet, here only implement the application of "Sparse alignment", "Dense alignment" and "Pose estimation". Trained model can be download [here](https://drive.google.com/open?id=14xWln1uqP13zxtUJ6B_aisYwWGdq53tP), please extract it into `./checkpoints`.

#### 1. Test on image
```bash
python test.py --img_path <testing_image_dir> 
```
Which will apply PRNet on images in the folder. Defult in `./Data/test-img`.

#### 2. Demo with webcam
```bash
python test.py --use_cam 
```
Which will apply PRNet on livestream webcam. Pressing `s` while testing will save the results to `save_path`, defult in `./Data/test-img`.
****


## References

Thanks for these source codes porviding me with knowledges to complete this repository.

- https://github.com/YadiraF/PRNet (Official)
    - PRNet architecture and API.
- https://github.com/YadiraF/face3d
    - Dataset generaton for position maps.
