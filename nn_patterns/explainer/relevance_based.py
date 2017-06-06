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
import lasagne.nonlinearities
import numpy as np
import theano
import theano.tensor as T

from .base import BaseInvertExplainer
from .gradient_based import GradientExplainer
from ..utils import misc as umisc


__all__ = [
    "LRPZExplainer",
    "LRPEpsExplainer",
]


class LRPZExplainer(GradientExplainer):

    def explain(self, X, target="max_output", **kwargs):
        return X * super(LRPZExplainer, self).explain(X,
                                                      target=target,
                                                      **kwargs)

    def get_name(self):
        return "lrp-z"

    def show_as_rgb(self):
        return True


class LRPEpsExplainer(BaseInvertExplainer):

    def __init__(self, *args, **kwargs):
        # Workaround for python 2 and 3.
        epsilon = kwargs.get("epsilon", 0.00000000001)
        if epsilon in kwargs:
            del kwargs["epsilon"]

        self.epsilon = epsilon
        super(LRPEpsExplainer, self).__init__(*args, **kwargs)

    def get_name(self):
        return "lrp-eps"

    def show_as_rgb(self):
        return False

    def _put_rectifiers(self, input_layer, layer):
        return umisc.get_rectifier_copy_layer(input_layer,layer)

    def _set_inverse_parameters(self, patterns=None):
        for l in L.get_all_layers(self.output_layer):
            if type(l) is L.Conv2DLayer:
                W = l.W.get_value()
                if l.flip_filters:
                    W = W[:,:,::-1,::-1]
                W = W.transpose(1,0,2,3)
                self.inverse_map[l].W.set_value(W)
            elif type(l) is L.DenseLayer:
                self.inverse_map[l].W.set_value(l.W.get_value().T)

    def _invert_LocalResponseNormalisation2DLayer(self, layer, feeder):
        return feeder

    def _get_normalised_relevance_layer(self, layer, feeder):

        def add_epsilon(Zs):
            tmp = (T.cast(Zs >= 0, theano.config.floatX)*2.0 - 1.0)
            return  Zs + self.epsilon * tmp

        if isinstance(layer, L.DenseLayer):
            forward_layer = L.DenseLayer(layer.input_layer,
                                         layer.num_units,
                                         W=layer.W,
                                         b=layer.b,
                                         nonlinearity=None)
        elif isinstance(layer, L.Conv2DLayer):
            forward_layer = L.Conv2DLayer(layer.input_layer,
                                          num_filters=layer.num_filters,
                                          W=layer.W,
                                          b=layer.b,
                                          stride=layer.stride,
                                          filter_size=layer.filter_size,
                                          flip_filters=layer.flip_filters,
                                          untie_biases=layer.untie_biases,
                                          pad=layer.pad,
                                          nonlinearity=None)
        else:
            raise NotImplementedError()

        forward_layer = L.ExpressionLayer(forward_layer,
                                          lambda x: 1.0 / add_epsilon(x))
        feeder = L.ElemwiseMergeLayer([forward_layer, feeder],
                                      merge_function=T.mul)

        return feeder

    def _invert_DenseLayer(self,layer,feeder):
        # Warning they are swapped here
        feeder = self._put_rectifiers(feeder, layer)
        feeder = self._get_normalised_relevance_layer(layer, feeder)

        output_units = np.prod(L.get_output_shape(layer.input_layer)[1:])
        output_layer = L.DenseLayer(feeder, num_units=output_units)
        W = output_layer.W
        tmp_shape = np.asarray((-1,)+L.get_output_shape(output_layer)[1:])
        x_layer = L.ReshapeLayer(layer.input_layer, tmp_shape.tolist())
        output_layer = L.ElemwiseMergeLayer(incomings=[x_layer, output_layer],
                                            merge_function=T.mul)
        output_layer.W = W
        return output_layer

    def _invert_Conv2DLayer(self,layer,feeder):
        # Warning they are swapped here
        feeder = self._put_rectifiers(feeder,layer)
        feeder = self._get_normalised_relevance_layer(layer,feeder)

        f_s = layer.filter_size
        if layer.pad == 'same':
            pad = 'same'
        elif layer.pad == 'valid' or layer.pad == (0, 0):
            pad = 'full'
        else:
            raise RuntimeError("Define your padding as full or same.")

        # By definition the
        # Flip filters must be on to be a proper deconvolution.
        num_filters = L.get_output_shape(layer.input_layer)[1]
        if layer.stride == (4,4):
            # Todo: similar code gradient based explainers. Merge.
            feeder = L.Upscale2DLayer(feeder, layer.stride, mode='dilate')
            output_layer = L.Conv2DLayer(feeder,
                                         num_filters=num_filters,
                                         filter_size=f_s,
                                         stride=1,
                                         pad=pad,
                                         nonlinearity=None,
                                         b=None,
                                         flip_filters=True)
            conv_layer = output_layer
            tmp = L.SliceLayer(output_layer, slice(0, -3), axis=3)
            output_layer = L.SliceLayer(tmp, slice(0, -3), axis=2)
            output_layer.W = conv_layer.W
        else:
            output_layer = L.Conv2DLayer(feeder,
                                         num_filters=num_filters,
                                         filter_size=f_s,
                                         stride=1,
                                         pad=pad,
                                         nonlinearity=None,
                                         b=None,
                                         flip_filters=True)
        W = output_layer.W

        # Do the multiplication.
        x_layer = L.ReshapeLayer(layer.input_layer,
                                 (-1,)+L.get_output_shape(output_layer)[1:])
        output_layer = L.ElemwiseMergeLayer(incomings=[x_layer, output_layer],
                                            merge_function=T.mul)
        output_layer.W = W
        return output_layer
