# [BIG-LITTLE NET: AN EFFICIENT MULTI-SCALE FEATURE REPRESENTATION FOR VISUAL AND SPEECH RECOGNITION](https://openreview.net/pdf?id=HJMHpjC9Ym)

## Introduction

This is an unofficial submission to ICLR 2019 Reproducibility Challenge. The central theme of the work by the authors is to reduce the computations while improving the accuracy in the case of Object Recognintion and Speech Recognition by using multiple branches with different scales in the CNN architecure. This helps in feature detection at different scales. The authors claim that in the case of Object Recognition they are able to improve the accuracy by 1% while reducing the computations by 1/3rd of the original.


## Checklist

- [ ] Getting the baseline from torchvision
- [ ] Testing the baseline
- [ ] Structure of the Project
- [ ] Building Big-Little Module
  - [ ] 1st half of the components
  - [ ] 2nd half of the components
- [ ] Injecting Resnet into Big-Little Net
- [ ] Run the models on GPUs
- [ ] Testing the reproducibility

## Progress

**5th March 2019** - Set-up repo, got ResNet baseline from [torchvision models](https://pytorch.org/docs/stable/torchvision/models.html). Figuring out details in Pytorch and covering some basics in it.

## Plan

**6th March 2019** - Think of project skeleton | Understand ResNet code

**7th March 2019** - Testing baseline

**8th March 2019** - Building Big-Little

**9th March 2019** - Building Big-Little

**10th March 2019** - Injecting Resnet into Big-Little Net
...


## Requirements

This repository uses:
- `Python 3.7`
- `PyTorch 1.0.1`

Using GPU is _highly_ recommended.

Recreate the environment using the following command.
```sh
conda create -e env.yml
```