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


import lasagne.layers as L
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor.nnet.neighbours

from .base import BasePatternComputer


__all__ = [
    "flatten",
    "get_dense_xy",
    "get_conv_xy",
    "get_conv_xy_all",
]


def flatten(W):
    """
    Get the flattened version of this weight matrix
    :param W:
    :return: W with D,O
    """
    if W.ndim==4:
        W = W.reshape(W.shape[0],-1)
        W = W.T
    return W


def get_dense_xy(layer, deterministic=True):
    x = L.get_output(L.FlattenLayer(layer.input_layer),
                     deterministic=deterministic)  # N, D
    w = layer.W # D, O
    y = T.dot(x, w)  # (N,O)
    if layer.b is not None:
        y += T.shape_padaxis(layer.b, axis=0)
    return x, y


def get_conv_xy(layer, deterministic=True):
    w_np = layer.W.get_value()
    input_layer = layer.input_layer
    if layer.pad == 'same':
        input_layer = L.PadLayer(layer.input_layer,
                                 width=np.array(w_np.shape[2:])/2,
                                 batch_ndim=2)
    input_shape = L.get_output_shape(input_layer)
    max_x = input_shape[2] - w_np.shape[2]
    max_y = input_shape[3] - w_np.shape[3]
    srng = RandomStreams()
    patch_x = srng.random_integers(low=0, high=max_x)
    patch_y = srng.random_integers(low=0, high=max_y)

    #print("input_shape shape: ", input_shape)
    #print("pad: \"%s\""% (layer.pad,))
    #print(" stride: " ,layer.stride)
    #print("max_x %d max_y %d"%(max_x,max_y))

    x = L.get_output(input_layer, deterministic=deterministic)
    x = x[:, :,
          patch_x:patch_x + w_np.shape[2], patch_y:patch_y + w_np.shape[3]]
    x = T.flatten(x, 2)  # N,D

    w = layer.W
    if layer.flip_filters:
        w = w[:, :, ::-1, ::-1]
    w = T.flatten(w, outdim=2).T  # D,O
    y = T.dot(x, w) # N,O
    if layer.b is not None:
        y += T.shape_padaxis(layer.b, axis=0)
    return x, y


def get_conv_xy_all(layer, deterministic=True):
    w_np = layer.W.get_value()
    w = layer.W
    if layer.flip_filters:
        w = w[:, :, ::-1, ::-1]

    input_layer = layer.input_layer
    if layer.pad == 'same':
        input_layer = L.PadLayer(layer.input_layer,
                                 width=np.array(w_np.shape[2:])//2,
                                 batch_ndim=2)
    input_shape = L.get_output_shape(input_layer)
    output_shape = L.get_output_shape(layer)
    max_x = input_shape[2] - w_np.shape[2]+1
    max_y = input_shape[3] - w_np.shape[3]+1
    #print("input_shape shape: ", input_shape)
    #print("output_shape shape: ", output_shape,np.prod(output_shape[2:]))
    #print("pad: \"%s\""%layer.pad)
    #print(" stride: " ,layer.stride)
    #print("max_x %d max_y %d"%(max_x,max_y))
    x_orig = L.get_output(input_layer, deterministic=True)

    x = theano.tensor.nnet.neighbours.images2neibs(x_orig,
                                                   neib_shape=layer.filter_size,
                                                   neib_step=layer.stride,
                                                   mode='valid')
    x = T.reshape(x, (x_orig.shape[0], -1,
                      np.prod(output_shape[2:]), np.prod(w_np.shape[2:])))
    x = T.transpose(x, (0, 2, 1, 3))
    x = T.reshape(x, (-1, T.prod(x.shape[2:])))

    w = T.flatten(w, outdim=2).T  # D,O
    y = T.dot(x, w) # N,O
    if layer.b is not None:
        y += T.shape_padaxis(layer.b, axis=0)
    return x, y
