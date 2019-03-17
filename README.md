# [BIG-LITTLE NET: AN EFFICIENT MULTI-SCALE FEATURE REPRESENTATION FOR VISUAL AND SPEECH RECOGNITION](https://openreview.net/pdf?id=HJMHpjC9Ym)

## Introduction

This is an unofficial submission to ICLR 2019 Reproducibility Challenge. The central theme of the work by the authors is to reduce the computations while improving the accuracy in the case of Object Recognition and Speech Recognition by using multiple branches with different scales in the CNN architecture. This helps in feature detection at different scales. The authors claim that in the case of Object Recognition they can improve the accuracy by 1% while reducing the computations by 1/3rd of the original.


## Checklist

- [x] Getting the resnet baseline from torchvision
- [x] Skeleton of the Project
- [x] Building Blocks
  - [x] `ResBlock`
  - [x] `ResBlockB`
  - [x] `ResBlockL`
  - [x] `TransitionLayer`
- [x] Getting runner code ready
- [x] Running baseline on slimmed data
- [x] Integrating Big-Little Net Blocks
- [x] Debug Issues
- [ ] Check correctness
- [ ] Run the models on GPUs
- [ ] Testing the reproducibility

## Progress

**5th March 2019** - Set-up repo, got ResNet baseline from [torchvision models][1]. Figuring out details in Pytorch and covering some basics in it.

**6th March 2019** - Pytorch Basics for CNNs | Understand ResNet code

**7th March 2019** - Plan the skeleton of implementation | Coded `Block` and `LayerDef` for BLNet which will help any architecture to be ported to Big-Little Net if is similar to ResNet. | Understand Inception Code to work out ways to implement `Branch`es in Big-Little Net.

**8th March 2019** - Setback: The original paper doesn't always follow specific guidelines for `num_branch > 2`. Therefore my approach to automating for `num_branch > 2` would not work. Currently only trying to make the automation work for `num_branch=1` and `num_branch=2`.

**9th March 2019** - Rethought the skeleton of the Project. | <_Setback_> | Prepared `ResBlock`, and it's children `ResBlockB`, `ResBlockL` and `TransitionLayer` for BL-Net.

_Setback_: The application approach needed the users to be informed of all the caveats of Big-Little Nets and its Network Architecture, threrfore beating the purpose of the generalized application for uninformed users.

**12th March 2019** - Got runner code for ImageNet from [Pytorch Examples][2] and ran `resnet18` on a slimmed dataset.

**14th March 2019** - Integrated Big-Little `Block`s. Running the code raises some assertions, need to check these.

**16th March 2019** - Corrected assersions, working to correct any error in architecture using [tensorboardX][3] for visualizing architecture. + worked on clearing some issues in architecture.


## Plan

**18th March 2019** - Resnet blocks when repeated don't have `stride = 2` at each block, need to remove that. Also, the paper mentions that `ResBlockB` uses a `stride` of `2` in the first `Conv3x3`. I think again a similar thing is happening, we only need to apply the stride in the first block, this makes sense too, as the big branch has `1/2` the image resolution than the little branch, therefore, there is no point it upsampling and downsampling the image dims inside the Big Branch itself.

**20th March 2019** - Try running on Distributed Environment. Reproduce the results for `bL-ResNet50`.


## Requirements

This repository uses:
- `Python 3.7`
- `PyTorch 1.0.1`

Using GPU is _highly_ recommended, the ImageNet dataset is nearly 160GBs, and the models are deep.

Recreate the environment using the following command.
```sh
conda create -n bln --file env.yml
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

Training is controlled by various options, which can be passed through the command line. The defaults can be looked at, in the file [``utils/options.py``](https://github.com/k0pch4/big-little-net/blob/master/helper/options.py). Try running below, for the help menu:
```sh
python3 train.py --help
```

## Citations

Please consider citing the original authors if you find the repository useful.

```
@article{DBLP:journals/corr/abs-1807-03848,
  author    = {Chun{-}Fu Chen and
               Quanfu Fan and
               Neil Mallinar and
               Tom Sercu and
               Rog{\'{e}}rio Schmidt Feris},
  title     = {Big-Little Net: An Efficient Multi-Scale Feature Representation for
               Visual and Speech Recognition},
  journal   = {CoRR},
  volume    = {abs/1807.03848},
  year      = {2018},
  url       = {http://arxiv.org/abs/1807.03848},
  archivePrefix = {arXiv},
  eprint    = {1807.03848},
  timestamp = {Mon, 13 Aug 2018 16:47:58 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1807-03848},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Code snippets taken from the following locations were extremely useful to be able to reproduce the results.

- [ResNet Model in PyTorch][1]
- [ImageNet Runner Code][2]
- [tensorboardX][3]

  [1]: https://pytorch.org/docs/stable/torchvision/models.html
  [2]: https://github.com/pytorch/examples/tree/master/imagenet
  [3]: https://github.com/lanpa/tensorboardX