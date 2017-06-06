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


__all__ = [
    "log_reg",

    "mlp_1dense",
    "mlp_2dense",

    "cnn_1convb_1dense",
    "cnn_2convb_1dense",
    "cnn_2convb_2dense",
    "cnn_3convb_2dense",
]


###############################################################################
###############################################################################
###############################################################################


def input_layer(*args, **kwargs):
    return lasagne.layers.InputLayer(*args, **kwargs)


def dense_layer(*args, **kwargs):
    return lasagne.layers.DenseLayer(*args, **kwargs) 


def conv_layer(*args, **kwargs):
    return lasagne.layers.Conv2DLayer(*args, **kwargs)


def conv_pool(layer_in, n_conv, prefix, n_filter, **kwargs):
    conv_prefix = "%s_%%i" % prefix

    ret = {}
    current_layer = layer_in
    for i in range(n_conv):
        conv = conv_layer(current_layer, n_filter,
                          (3, 3), (1, 1), pad="same", 
                          W=lasagne.init.GlorotUniform(), **kwargs)
        current_layer = conv
        ret[conv_prefix % i] = conv

    ret["%s_pool" % prefix] = lasagne.layers.MaxPool2DLayer(current_layer,
                                                            (2, 2))
    return ret


def dropout_layer(*args, **kwargs):
    return lasagne.layers.DropoutLayer(*args, **kwargs)


###############################################################################
###############################################################################
###############################################################################


def log_reg(input_shape, output_n, nonlinearity=None):
    if nonlinearity is None:
        nonlinearity = lasagne.nonlinearities.rectify

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net["out"] = dense_layer(net["in"], num_units=output_n,
                             nonlinearity=lasagne.nonlinearities.softmax,
                             W=lasagne.init.GlorotUniform())

    net.update({
        "input_shape": input_shape,
        "input_var": net["in"].input_var,

        "output_n": output_n,
    })
    return net


###############################################################################
###############################################################################
###############################################################################


def mlp_1dense(input_shape, output_n, nonlinearity=None,
               dense_units=512, dropout_rate=0.25):
    if nonlinearity is None:
        nonlinearity = lasagne.nonlinearities.rectify

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net["dense_1"] = dense_layer(net["in"], num_units=dense_units,
                                 nonlinearity=nonlinearity,
                                 W=lasagne.init.GlorotUniform())
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], p=dropout_rate)
    net["out"] = dense_layer(net["dense_1_dropout"], num_units=output_n,
                             nonlinearity=lasagne.nonlinearities.softmax,
                             W=lasagne.init.GlorotUniform())

    net.update({
        "input_shape": input_shape,
        "input_var": net["in"].input_var,

        "output_n": output_n,
    })
    return net


def mlp_2dense(input_shape, output_n, nonlinearity=None,
               dense_units=512, dropout_rate=0.25):
    if nonlinearity is None:
        nonlinearity = lasagne.nonlinearities.rectify

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net["dense_1"] = dense_layer(net["in"], num_units=dense_units,
                                 nonlinearity=nonlinearity,
                                 W=lasagne.init.GlorotUniform())
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], p=dropout_rate)
    net["dense_2"] = dense_layer(net["dense_1_dropout"], num_units=dense_units,
                                 nonlinearity=nonlinearity,
                                 W=lasagne.init.GlorotUniform())
    net['dense_2_dropout'] = dropout_layer(net['dense_2'], p=dropout_rate)
    net["out"] = dense_layer(net["dense_2_dropout"], num_units=output_n,
                             nonlinearity=lasagne.nonlinearities.softmax,
                             W=lasagne.init.GlorotUniform())

    net.update({
        "input_shape": input_shape,
        "input_var": net["in"].input_var,

        "output_n": output_n,
    })
    return net


###############################################################################
###############################################################################
###############################################################################


def cnn_1convb_1dense(input_shape, output_n, nonlinearity=None,
                      dense_units=512, dropout_rate=0.25):
    if nonlinearity is None:
        nonlinearity = lasagne.nonlinearities.rectify

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net.update(conv_pool(net["in"], 2, "conv_1", 128,
                         nonlinearity=nonlinearity))
    net["dense_1"] = dense_layer(net["conv_1_pool"], num_units=dense_units,
                                 nonlinearity=nonlinearity,
                                 W=lasagne.init.GlorotUniform())
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], p=dropout_rate)
    net["out"] = dense_layer(net["dense_1_dropout"], num_units=output_n,
                             nonlinearity=lasagne.nonlinearities.softmax,
                             W=lasagne.init.GlorotUniform())

    net.update({
        "input_shape": input_shape,
        "input_var": net["in"].input_var,

        "output_n": output_n,
    })
    return net


def cnn_2convb_1dense(input_shape, output_n, nonlinearity=None,
                      dense_units=512, dropout_rate=0.25):
    if nonlinearity is None:
        nonlinearity = lasagne.nonlinearities.rectify

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net.update(conv_pool(net["in"], 2, "conv_1", 128,
                         nonlinearity=nonlinearity))
    net.update(conv_pool(net["conv_1_pool"], 2, "conv_2", 128,
                         nonlinearity=nonlinearity))
    net["dense_1"] = dense_layer(net["conv_2_pool"], num_units=dense_units,
                                 nonlinearity=nonlinearity,
                                 W=lasagne.init.GlorotUniform())
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], p=dropout_rate)
    net["out"] = dense_layer(net["dense_1_dropout"], num_units=output_n,
                             nonlinearity=lasagne.nonlinearities.softmax,
                             W=lasagne.init.GlorotUniform())

    net.update({
        "input_shape": input_shape,
        "input_var": net["in"].input_var,

        "output_n": output_n,
    })
    return net


def cnn_2convb_2dense(input_shape, output_n, nonlinearity=None,
                      dense_units=512, dropout_rate=0.25):
    if nonlinearity is None:
        nonlinearity = lasagne.nonlinearities.rectify

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net.update(conv_pool(net["in"], 2, "conv_1", 128,
                         nonlinearity=nonlinearity))
    net.update(conv_pool(net["conv_1_pool"], 2, "conv_2", 128,
                         nonlinearity=nonlinearity))
    net["dense_1"] = dense_layer(net["conv_2_pool"], num_units=dense_units,
                                 nonlinearity=nonlinearity,
                                 W=lasagne.init.GlorotUniform())
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], p=dropout_rate)
    net["dense_2"] = dense_layer(net["dense_1_dropout"], num_units=dense_units,
                                 nonlinearity=nonlinearity,
                                 W=lasagne.init.GlorotUniform())
    net['dense_2_dropout'] = dropout_layer(net['dense_2'], p=dropout_rate)
    net["out"] = dense_layer(net["dense_2_dropout"], num_units=output_n,
                             nonlinearity=lasagne.nonlinearities.softmax,
                             W=lasagne.init.GlorotUniform())

    net.update({
        "input_shape": input_shape,
        "input_var": net["in"].input_var,

        "output_n": output_n,
    })
    return net


def cnn_3convb_2dense(input_shape, output_n, nonlinearity=None,
                      dense_units=512, dropout_rate=0.25):
    if nonlinearity is None:
        nonlinearity = lasagne.nonlinearities.rectify

    net = {}
    net["in"] = input_layer(shape=input_shape)
    net.update(conv_pool(net["in"], 2, "conv_1", 128,
                         nonlinearity=nonlinearity))
    net.update(conv_pool(net["conv_1_pool"], 2, "conv_2", 128,
                         nonlinearity=nonlinearity))
    net.update(conv_pool(net["conv_2_pool"], 2, "conv_3", 128,
                         nonlinearity=nonlinearity))
    net["dense_1"] = dense_layer(net["conv_3_pool"], num_units=dense_units,
                                 nonlinearity=nonlinearity,
                                 W=lasagne.init.GlorotUniform())
    net['dense_1_dropout'] = dropout_layer(net['dense_1'], p=dropout_rate)
    net["dense_2"] = dense_layer(net["dense_1_dropout"], num_units=dense_units,
                                 nonlinearity=nonlinearity,
                                 W=lasagne.init.GlorotUniform())
    net['dense_2_dropout'] = dropout_layer(net['dense_2'], p=dropout_rate)
    net["out"] = dense_layer(net["dense_2_dropout"], num_units=output_n,
                             nonlinearity=lasagne.nonlinearities.softmax,
                             W=lasagne.init.GlorotUniform())

    net.update({
        "input_shape": input_shape,
        "input_var": net["in"].input_var,

        "output_n": output_n,
    })
    return net
