# [BIG-LITTLE NET: AN EFFICIENT MULTI-SCALE FEATURE REPRESENTATION FOR VISUAL AND SPEECH RECOGNITION](https://openreview.net/pdf?id=HJMHpjC9Ym)

## Introduction

This is an unofficial submission to ICLR 2019 Reproducibility Challenge. The central theme of the work by the authors is to reduce the computations while improving the accuracy in the case of Object Recognition and Speech Recognition by using multiple branches with different scales in the CNN architecture. This helps in feature detection at different scales. The authors claim that in the case of Object Recognition they can improve the accuracy by 1% while reducing the computations by 1/3rd of the original.


## Checklist

- [x] Getting the resnet baseline from torchvision
- [x] Skeleton of the Project
- [ ] Building Blocks
  - [ ] `ResBlock`
  - [ ] `ResBlockB`
  - [ ] `ResBlockL`
  - [ ] `TransitionLayer`
- [ ] Integrating `Block`s
- [ ] Testing the baseline
- [ ] Injecting Resnet into Big-Little Net
- [ ] Run the models on GPUs
- [ ] Testing the reproducibility

## Progress

**5th March 2019** - Set-up repo, got ResNet baseline from [torchvision models](https://pytorch.org/docs/stable/torchvision/models.html). Figuring out details in Pytorch and covering some basics in it.

**6th March 2019** - Pytorch Basics for CNNs | Understand ResNet code

**7th March 2019** - Plan the skeleton of implementation | Coded `Block` and `LayerDef` for BLNet which will help any architecture to be ported to Big-Little Net if is similar to ResNet. | Understand Inception Code to work out ways to implement `Branch`es in Big-Little Net.

**8th March 2019** - Setback: The original paper doesn't always follow specific guidelines for `num_branch > 2`. Therefore my approach to automating for `num_branch > 2` would not work. Currently only trying to make the automation work for `num_branch=1` and `num_branch=2`.

**9th March 2019** - Rethought the skeleton of the Project. | <_Setback_> | Retry from scratch.

_Setback_: The application approach needed the users to be informed of all the caveats of Big-Little Nets and its Network Architecture, threrfore beating the purpose of the generalized application for uninformed users.

## Plan

**10th March 2019** - Work on Implement simplified variant


## Requirements

This repository uses:
- `Python 3.7`
- `PyTorch 1.0.1`

Using GPU is _highly_ recommended.

Recreate the environment using the following command.
```sh
conda create -e env.yml
```

## Scope

The scope of this reproducibility challenge is to reproduce the table given below.

| Model                   | Top-1 Error (%) |
|-------------------------|-----------------|
| Resnet-50               | 23.66           |
| bL-Resnet-50 (a=2, b=2) | 22.72           |
| bL-Resnet-50 (a=2, b=4) | 22.69           |
| bL-Resnet-50 (a=4, b=2) | 23.20           |
| bL-Resnet-50 (a=4, b=2) | 23.15           |

The Network architecture for bL-Resnet-50:

![](https://i.imgur.com/mQ3M5T0.png)


## Reproduced Results


## Training

Training is controlled by various options, which can be passed through the command line. The defaults can be looked at, in the file [``utils/options.py``](https://github.com/k0pch4/big-little-net/blob/master/utils/options.py). Try running for the help menu:
```sh
python3 train.py --help
```
