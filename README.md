
This repository hosts the source code of our paper: [[ACCV 2022] PS-ARM: An End-to-End Attention-aware Relation Mixer Network for Person Search](https://arxiv.org/abs/2210.03433). 

## Abstract

Person search is a challenging problem with various real-world applications, that aims at joint person detection and re-identification of a query person from uncropped gallery images. Although, previous study focuses on rich feature information learning, it’s still hard to retrieve the query person due to the occurrence of appearance deformations and background distractors. In this paper, we propose a novel attention-aware relation mixer (ARM) module for person search, which exploits the global relation between different local regions within RoI of a person and make it robust against various appearance deformations and occlusion. The proposed ARM is composed of a relation mixer block and a spatio-channel attention layer. The relation mixer block introduces a spatially attended spatial mixing and a channel-wise attended channel mixing for effectively capturing discriminative relation features within an RoI. These discriminative relation features are further enriched by intro-
ducing a spatio-channel attention where the foreground and background discriminability is empowered in a joint spatio-channel space. Our ARM module is generic and it does not rely on fine-grained supervisions or topological assumptions, hence being easily integrated into any Faster R-CNN based person search methods. Comprehensive experiments are performed on two challenging benchmark datasets: CUHK-SYSU and PRW. Our PS-ARM achieves state-of-the-art performance on both datasets. On the challenging PRW dataset, our PS-ARM achieves an absolute gain of 5% in the mAP score over SeqNet, while operating at a comparable speed. 


Performance profile:

| Dataset   | mAP  | Top-1 | Model                                                        |
| --------- | ---- | ----- | ------------------------------------------------------------ |
| CUHK-SYSU | 95.2 | 96.1  | [model](https://drive.google.com/file/d/1G1CmnLukVoWhUwuxIzl6LN7Ck1VoJ4TB/view?usp=sharing) |
| PRW       | 52.6 | 88.1  | [model](https://drive.google.com/file/d/1LAILssRq_NctoWtPKjuRIK4PM2bx_j9N/view?usp=sharing) |


## Installation

Create the environment using yml  `conda env create -f armnet.yml` in the root directory of the project.


## Quick Start

Let's say `$ROOT` is the root directory.

1. Download [CUHK-SYSU](https://drive.google.com/open?id=1z3LsFrJTUeEX3-XjSEJMOBrslxD2T5af) and [PRW](https://goo.gl/2SNesA) datasets, and unzip them to `$ROOT/data`
```
$ROOT/data
├── CUHK-SYSU
└── PRW
```
2. Following the link in the above table, download our pretrained model to anywhere you like, e.g., `$ROOT/exp_cuhk`
3. Run an inference demo by specifing the paths of checkpoint and corresponding configuration file. `python train.py --cfg $ROOT/exp_cuhk/config.yaml --ckpt $ROOT/exp_cuhk/best_cuhk.ph` You can checkout the result in `demo_imgs` directory.

![demo.jpg](./demo_imgs/demo.jpg)

## Training

Pick one configuration file you like in `$ROOT/configs`, and run with it.

```
python train.py --cfg configs/cuhk_sysu.yaml
```

**Note**: At present, our script only supports single GPU training, but distributed training will be also supported in future. By default, the batch size and the learning rate during training are set to 3 and 0.003 respectively. If your GPU cannot provide the required memory, try smaller batch size and learning rate (*performance may degrade*). Specifically, your setting should follow the [*Linear Scaling Rule*](https://arxiv.org/abs/1706.02677): When the minibatch size is multiplied by k, multiply the learning rate by k. For example:

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
python train.py --cfg $ROOT/exp_cuhk/config.yaml --eval --ckpt $ROOT/exp_cuhk/epoch_xx.pth 
```

Test the upper bound of the person search performance by using GT boxes:

```
python train.py --cfg $ROOT/exp_cuhk/config.yaml --eval --ckpt $ROOT/exp_cuhk/epoch_xx.pth EVAL_USE_GT True
```


## Citation

```
@inproceedings{fiaz2022psarm,
  title={PS-ARM: An End-to-End Attention-aware Relation Mixer Network for Person Search},
  author={Fiaz, Mustansar and Cholakkal, Hisham and Narayan, Sanath and Anwar, Rao Muhammad and Khan, Fahad Shahbaz},
  booktitle={Proceedings of the ACCV Asian Conference on Computer Vision}, 
  year={2022}
}
```

## Contact
Should you have any question, please create an issue on this repository or contact at mustansar.fiaz@mbzuai.ac.ae

<hr />

## References
Our code is based on [SeqNet](https://github.com/serend1p1ty/SeqNet) repository. 
We thank them for releasing their baseline code.
* **PS-ARM**: "PS-ARM: An End-to-End Attention-aware Relation Mixer Network for Person Search", ACCV, 2022 (*MBZUAI*). [[Paper](https://arxiv.org/abs/2210.03433)][[PyTorch](https://github.com/mustansarfiaz/PS-ARM)]
