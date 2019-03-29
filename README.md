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
- [x] Check correctness
- [x] Added basic tests
- [x] Add Nesterov SDG with Cosine LR Scheduler to the runner code
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

**17th March 2019** - Resnet blocks when repeated don't have `stride = 2` at each block, need to remove that. Also, the paper mentions that `ResBlockB` uses a `stride` of `2` in the first `Conv3x3`. I think again a similar thing is happening, we only need to apply the stride in the first block, this makes sense too, as the big branch has `1/2` the image resolution than the little branch, therefore, there is no point it upsampling and downsampling the image dims inside the Big Branch itself (authors were really supportive and confirmed that upsampling happens at the end, before the merging of the two branches. ~~Every `ResBlockB` has a `stride = 3` for the conv3x3 and every one of it ends with upsampling, read in the paper.~~

**18-21th March 2019** - Waiting for GPU access.

**22th March 2019** - Correct the implementation by having upsampling at the end of the branches itself.

**25th March 2019** - Running on 8 Nvidia-V100 16GB GPUs, taking `batch_size=1024` due to time and money contraint. Taking batch size as 1024 as it is the fastest I can go on 16GB cards (according to the idea that batch sizes should be multiple of 2s). Also using `lr=0.4` according to the results by the paper [Accurate, Large Minibatch SGD:
Training ImageNet in 1 Hour][4].

**29th March 2019** - Added basic tests. Reduced memory usage by removing initilization of upsampling convs for `ResBlockL` other than the last block in blModule. Added Cosine Scheduler, also, the period of the cosine annealing is set to 1 by the authors, thus implicitly having no restarts.


## Plan

**xx** - Run the model on a smaller dataset and try to see if any errors pertain further after that try to reproduce the results for `bL-ResNet50`.


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

xxx


## Training

[`train.py`][https://github.com/k0pch4/big-little-net/blob/master/train.py] below assumes that the ImageNet dataset path passed contains 2 folders, `train` and `val`. You could use the script [`valprep.sh`][6] to move the images from `val` in the corresponding labeled subfolders.

Training is controlled by various options, which can be passed through the command line. The defaults can be looked at, in the file [``utils/options.py``](https://github.com/k0pch4/big-little-net/blob/master/helper/options.py). Defaults are set such that we would be required to set them unless we want to. Below is a sample run:
```sh
python3 train.py .imagenet/ --epochs 4 --lr 0.1 --alpha 2 --beta 4 --workers 4 -a bl_resnet50
```


## Architecture Visualization

To have a look at the architecture you can use the following command to generate tensorboard files (by default in `./run`) to view the architecture.
```sh
python3 viz.py
tensorboard --port 8888 --logdir runs
```

visit **`localhost:8888`** to view the architecture. Look at `./arch.png` if you can't run tensorboad.


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

To train on large batch sizes (train faster) we need to change the learning rate as to maintain the accuracy of the network, therefore I am using `lr=4*old_lr=0.4` since the original batch size is 1/4th of the batch size I used. Citations for the relevant work below:

```
@article{DBLP:journals/corr/GoyalDGNWKTJH17,
  author    = {Priya Goyal and
               Piotr Doll{\'{a}}r and
               Ross B. Girshick and
               Pieter Noordhuis and
               Lukasz Wesolowski and
               Aapo Kyrola and
               Andrew Tulloch and
               Yangqing Jia and
               Kaiming He},
  title     = {Accurate, Large Minibatch {SGD:} Training ImageNet in 1 Hour},
  journal   = {CoRR},
  volume    = {abs/1706.02677},
  year      = {2017},
  url       = {http://arxiv.org/abs/1706.02677},
  archivePrefix = {arXiv},
  eprint    = {1706.02677},
  timestamp = {Mon, 13 Aug 2018 16:49:10 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/GoyalDGNWKTJH17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Code snippets taken from the following locations were extremely useful to be able to reproduce the results.

- [ResNet Model in PyTorch][1]
- [ImageNet Runner Code][2]
- [tensorboardX][3]
- [torchtest][5]

  [1]: https://pytorch.org/docs/stable/torchvision/models.html
  [2]: https://github.com/pytorch/examples/tree/master/imagenet
  [3]: https://github.com/lanpa/tensorboardX
  [4]: https://arxiv.org/pdf/1706.02677.pdf
  [5]: https://github.com/suriyadeepan/torchtest
  [6]: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
