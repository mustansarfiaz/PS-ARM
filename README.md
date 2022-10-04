
This repository hosts the source code of our paper: [[ACCV 2022]PS-ARM: An End-to-End Attention-aware Relation Mixer Network for Person Search](https://arxiv.org/abs/2103.10148). 

Performance profile:

| Dataset   | mAP  | Top-1 | Model                                                        |
| --------- | ---- | ----- | ------------------------------------------------------------ |
| CUHK-SYSU | 94.8 | 95.7  | [model](https://drive.google.com/file/d/1wKhCHy7uTHx8zxNS62Y1236GNv5TzFzq/view?usp=sharing) |
| PRW       | 47.6 | 87.6  | [model](https://drive.google.com/file/d/1I9OI6-sfVyop_aLDIWaYwd7Z4hD34hwZ/view?usp=sharing) |


## Installation

Create the environment using yml  `conda env create -f environment.yml` in the root directory of the project.


## Quick Start

Let's say `$ROOT` is the root directory.

1. Download [CUHK-SYSU](https://drive.google.com/open?id=1z3LsFrJTUeEX3-XjSEJMOBrslxD2T5af) and [PRW](https://goo.gl/2SNesA) datasets, and unzip them to `$ROOT/data`
```
$ROOT/data
├── CUHK-SYSU
└── PRW
```
2. Following the link in the above table, download our pretrained model to anywhere you like, e.g., `$ROOT/exp_cuhk`
3. Run an inference demo by specifing the paths of checkpoint and corresponding configuration file. `python train.py --cfg $ROOT/exp_cuhk/config.yaml --ckpt $ROOT/exp_cuhk/epoch_19.pth` You can checkout the result in `demo_imgs` directory.

![demo.jpg](./demo_imgs/demo.jpg)

## Training

Pick one configuration file you like in `$ROOT/configs`, and run with it.

```
python train.py --cfg configs/cuhk_sysu.yaml
```

**Note**: At present, our script only supports single GPU training, but distributed training will be also supported in future. By default, the batch size and the learning rate during training are set to 5 and 0.003 respectively, which requires about 28GB of GPU memory. If your GPU cannot provide the required memory, try smaller batch size and learning rate (*performance may degrade*). Specifically, your setting should follow the [*Linear Scaling Rule*](https://arxiv.org/abs/1706.02677): When the minibatch size is multiplied by k, multiply the learning rate by k. For example:

```
python train.py --cfg configs/cuhk_sysu.yaml INPUT.BATCH_SIZE_TRAIN 3 SOLVER.BASE_LR 0.0003
```

**Tip**: If the training process stops unexpectedly, you can resume from the specified checkpoint.

```
python train.py --cfg configs/cuhk_sysu.yaml --resume --ckpt /path/to/your/checkpoint
```

## Test

Suppose the output directory is `$ROOT/exp_cuhk`. Test the trained model:

```
python train.py --cfg $ROOT/exp_cuhk/config.yaml --eval --ckpt $ROOT/exp_cuhk/epoch_10.pth EVAL_USE_CBGM True
```

Test the upper bound of the person search performance by using GT boxes:

```
python train.py --cfg $ROOT/exp_cuhk/config.yaml --eval --ckpt $ROOT/exp_cuhk/epoch_10.pth EVAL_USE_GT True
```


## Citation

```
@inproceedings{fiaz2022psarm,
  title={PS-ARM: An End-to-End Attention-aware Relation Mixer Network for Person Search},
  author={Fiaz, Mustansar and Cholakkal, Hisham and Narayan, Sanath and Anwar, Muhammad Rao and Khan, Fahad Shahbaz},
  booktitle={Proceedings of the ACCV Asian Conference on Computer Vision}, 
  year={2022}
}
```
