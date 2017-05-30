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


###############################################################################
###############################################################################
###############################################################################


import lasagne.init
import lasagne.layers
import lasagne.nonlinearities
import numpy as np
import theano

from . import base


__all__ = [
    "vgg16",
    #"vgg16_all_conv",

    #"caffenet",
    #"googlenet",
]


###############################################################################
###############################################################################
###############################################################################


def vgg16(nonlinearity):
    input_shape = [None, 3, 224, 224]
    output_n = 1000
    
    net = {}
    net["in"] = base.input_layer(shape=input_shape)

    net.update(base.conv_pool(net["in"], 2, "conv_1", 64,
                              nonlinearity=nonlinearity))
    net.update(base.conv_pool(net["conv_1_pool"], 2, "conv_2", 128,
                              nonlinearity=nonlinearity))
    net.update(base.conv_pool(net["conv_2_pool"], 3, "conv_3", 256,
                              nonlinearity=nonlinearity))
    net.update(base.conv_pool(net["conv_3_pool"], 3, "conv_4", 512,
                              nonlinearity=nonlinearity))
    net.update(base.conv_pool(net["conv_4_pool"], 3, "conv_5", 512,
                              nonlinearity=nonlinearity))
    
    net["dense_1"] = base.dense_layer(net["conv_5_pool"], num_units=4096,
                                      nonlinearity=nonlinearity,
                                      W=lasagne.init.GlorotUniform())
    net['dense_1_dropout'] = base.dropout_layer(net['dense_1'], p=0.5)
    net["dense_2"] = base.dense_layer(net["dense_1_dropout"], num_units=4096,
                                      nonlinearity=nonlinearity,
                                      W=lasagne.init.GlorotUniform())
    net['dense_2_dropout'] = base.dropout_layer(net['dense_2'], p=0.5)
    net["out"] = base.dense_layer(net["dense_2_dropout"], num_units=output_n,
                                  nonlinearity=lasagne.nonlinearities.softmax,
                                  W=lasagne.init.GlorotUniform())

    net.update({
        "input_shape": input_shape,
        "input_var": net["in"].input_var,

        "output_n": output_n,
    })
    return net


def vgg16_all_conv(nonlinearity):
    input_shape = [None, 3, 256, 256]
    output_n = 1000
    
    net = {}
    net["in"] = base.input_layer(shape=input_shape)

    net.update(base.conv_pool(net["in"], 2, "conv_1", 64,
                              nonlinearity=nonlinearity))
    net.update(base.conv_pool(net["conv_1_pool"], 2, "conv_2", 128,
                              nonlinearity=nonlinearity))
    net.update(base.conv_pool(net["conv_2_pool"], 3, "conv_3", 256,
                              nonlinearity=nonlinearity))
    net.update(base.conv_pool(net["conv_3_pool"], 3, "conv_4", 512,
                              nonlinearity=nonlinearity))
    net.update(base.conv_pool(net["conv_4_pool"], 3, "conv_5", 512,
                              nonlinearity=nonlinearity))
    
    net["dense_1"] = base.conv_layer(net["conv_5_pool"],
                                     4096, 7, pad="valid",
                                     flip_filters=False,
                                     nonlinearity=nonlinearity,
                                     W=lasagne.init.GlorotUniform())
    net['dense_1_dropout'] = base.dropout_layer(net['dense_1'], p=0.5)
    net["dense_2"] = base.conv_layer(net["dense_1_dropout"],
                                     4096, 1, pad="valid",
                                     flip_filters=False,
                                     nonlinearity=nonlinearity,
                                     W=lasagne.init.GlorotUniform())
    net['dense_2_dropout'] = base.dropout_layer(net['dense_2'], p=0.5)
    net["dense_3"] = base.conv_layer(net["dense_2_dropout"], output_n, 1,
                                     pad="valid", flip_filters=False,
                                     nonlinearity=None,
                                     W=lasagne.init.GlorotUniform())
    net["global_pool"] = lasagne.layers.GlobalPoolLayer(
        net["dense_3"], pool_function=theano.tensor.sum)
    net["out"] = lasagne.layers.NonlinearityLayer(
        net["global_pool"], nonlinearity=lasagne.nonlinearities.softmax)

    net.update({
        "input_shape": input_shape,
        "input_var": net["in"].input_var,

        "output_n": output_n,
    })
    return net
