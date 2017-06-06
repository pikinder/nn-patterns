# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os

import nn_patterns
import nn_patterns.utils.fileio
import nn_patterns.utils.tests.networks.imagenet
import lasagne
import theano

import imp
base_dir = os.path.dirname(__file__)
eutils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))

param_file = "./imagenet_224_vgg_16.npz"
# Note those weights are CC 4.0:
# See http://www.robots.ox.ac.uk/~vgg/research/very_deep/
param_url = "https://www.dropbox.com/s/cvjj8x19hzya9oe/imagenet_224_vgg_16.npz?dl=1"

pattern_file = "./imagenet_224_vgg_16.pattern_file.A_only.npz"
pattern_url = "https://www.dropbox.com/s/v7e0px44jqwef5k/imagenet_224_vgg_16.patterns.A_only.npz?dl=1"

if __name__ == "__main__":
    # Download the necessary parameters for VGG16 and the according patterns.
    eutils.download(param_url, param_file)
    eutils.download(pattern_url, pattern_file)

    # Get some example test set images.
    images, label_to_class_name = eutils.get_imagenet_data()
    image = images[0][0]

    ###########################################################################
    # Build model.
    ###########################################################################
    parameters = nn_patterns.utils.fileio.load_parameters(param_file)
    vgg16 = nn_patterns.utils.tests.networks.imagenet.vgg16()
    lasagne.layers.set_all_param_values(vgg16["out"], parameters)

    # Create prediction model.
    predict_f = theano.function([vgg16["input_var"]],
                                lasagne.layers.get_output(vgg16["out"],
                                                          deterministic=True))


    ###########################################################################
    # Explanations.
    ###########################################################################
    # Lets use pretrained patterns.
    # On how to train patterns, please see the cifar10 example.
    patterns = nn_patterns.utils.fileio.load_patterns(pattern_file)

    # Create explainers.
    pattern_net_explainer = nn_patterns.create_explainer("patternnet",
                                                         vgg16["out"],
                                                         patterns=[patterns])
    pattern_lrp_explainer = nn_patterns.create_explainer("patternlrp",
                                                         vgg16["out"],
                                                         patterns=[patterns])
    # Create explanations.
    x = eutils.preprocess(image)[None, :, :, :]
    target = "max_output"  # Explain output neuron with max activation.

    pattern_net_exp = pattern_net_explainer.explain(x, target=target)[0]
    pattern_lrp_exp = pattern_lrp_explainer.explain(x, target=target)[0]

    # Postprocess.
    pattern_net_exp = eutils.back_projection(pattern_net_exp)
    pattern_lrp_exp = eutils.heatmap(pattern_lrp_exp)


    ###########################################################################
    # Plot the explanations.
    ###########################################################################
    plt.clf()
    fig, axs = plt.subplots(1, 3)
    for i in range(3):
        axs[i].tick_params(axis="x", which="both",
                           bottom="off", top="off", labelbottom="off")
        axs[i].tick_params(axis="y", which="both",
                           bottom="off", top="off", labelbottom="off")
        axs[i].axis("off")


    axs[0].imshow(image[:, :, ::-1] / 255,
                     interpolation="nearest")
    axs[1].imshow(pattern_net_exp.transpose(1, 2, 0),
                     interpolation="nearest")
    axs[2].imshow(pattern_lrp_exp.transpose(1, 2, 0),
                     interpolation="nearest")

    plt.savefig("step_by_step_imagenet.pdf")
