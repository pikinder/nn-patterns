# PatternNet, PatternLRP and more


## Introduction

PatternNet and PatternLRP are methods that help to interpret decision of non-linear neural networks.
They are in a line with the methods DeConvNet, GuidedBackprop and LRP:

![An overview of the different explanation methods.](https://raw.githubusercontent.com/pikinder/nn-patterns/master/images/fig2.png)

and improve on them:

![Different explanation methods on ImageNet.](https://raw.githubusercontent.com/pikinder/nn-patterns/master/images/fig5.png)

For more details we refer to the paper:

```
PatternNet and PatternLRP -- Improving the interpretability of neural networks
Pieter-Jan Kindermans, Kristof T. Schütt, Maximilian Alber, Klaus-Robert Müller, Sven Dähne
https://arxiv.org/abs/1705.05598
```

If you use this code please cite the following paper:
```
TODO: Add link to SW paper.
```


## Installation

To install the code, please clone the repository and run the setup script:

```bash
git clone https://github.com/pikinder/nn-patterns.git
cd nn-patterns
python setup.py install
```

## Usage and Examples

TODO.

In the directory 'examples' you find different examples as Python scripts and as Jupyter notebooks:

* step_by_step_cifar10: explains how to compute patterns for a given neural networks and how to use them with PatternNet and PatternLRP.
* step_by_step_imagenet: explains how to apply pre-computed patterns for the VGG16 network on ImageNet.
* all_methods: shows how to use the different methods with VGG16 on ImageNet, i.e. the reproduce the explanation grid above.
