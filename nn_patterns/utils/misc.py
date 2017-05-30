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


import contextlib
import lasagne.layers
import lasagne.nonlinearities
from lasagne.nonlinearities import rectify
import theano.tensor as T


@contextlib.contextmanager
def ignore_sigmoids(layer):
    if(hasattr(layer,'nonlinearity') and
       layer.nonlinearity in [lasagne.nonlinearities.softmax,
                              lasagne.nonlinearities.sigmoid]):
        print("Removing the sigmoids from output for the explanation approach.")
        nonlinearity = layer.nonlinearity
        layer.nonlinearity = lambda x: x
        try:
            yield layer
        finally:
            layer.nonlinearity = nonlinearity
    else:
        yield layer


def remove_sigmoids(layer):
    if(hasattr(layer,'nonlinearity') and
       layer.nonlinearity in [lasagne.nonlinearities.softmax,
                              lasagne.nonlinearities.sigmoid]):
        layer.nonlinearity = lambda x: x


class GuidedReLU(lasagne.layers.MergeLayer):
    """
    A layer with two input streams of which the
    first will be passed on as long as the
    second is not 0. If the second input stream
    is not None, then it will be passed on
    instead.
    """

    def __init__(self, input_layer, other_layer):
        super(GuidedReLU, self).__init__([input_layer, other_layer])

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        in1, in2 = inputs
        out = T.switch(T.gt(in2, 0.), in1, 0.)
        return out


class OppositeGuidedRelu(lasagne.layers.MergeLayer):
    """
    A layer with two input streams of which the
    first will be passed on as long as the
    second is 0. If the second input stream is
    not None, then it will be passed on
    instead.
    """

    def __init__(self, input_layer, other_layer):
        super(OppositeGuidedRelu, self).__init__([input_layer, other_layer])

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        in1, in2 = inputs
        out = T.switch(T.gt(in2, 0.), 0., in1)
        return out


def has_ReLU(layer):
    relus = [lasagne.nonlinearities.rectify, T.nnet.relu]
    return (hasattr(layer, 'nonlinearity') and
            layer.nonlinearity in relus)


def get_rectifier_copy_layer(input_layer, rectifier_layer):
    if has_ReLU(rectifier_layer):
        return GuidedReLU(input_layer, rectifier_layer)
    return input_layer


def get_rectifier_opposite_layer(input_layer, rectifier_layer):
    if has_ReLU(rectifier_layer):
        return OppositeGuidedRelu(input_layer, rectifier_layer)
    return None


def get_rectifier_layer(input_layer, rectifier_layer):
    if has_ReLU(rectifier_layer):
        return lasagne.layers.NonlinearityLayer(input_layer,
                                                nonlinearity=rectify)
    return input_layer
