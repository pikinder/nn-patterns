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
from ..utils import misc as umisc


__all__ = [
    "PatternNetExplainer",
    "GuidedPatternNetExplainer",
    "PatternLRPExplainer",
]


class PatternNetExplainer(BaseInvertExplainer):

    def _invert_LocalResponseNormalisation2DLayer(self, layer, feeder):
        return feeder

    def _set_inverse_parameters(self, patterns):
        self.trainable_layers = [self.inverse_map[l]
                                 for l in L.get_all_layers(self.output_layer)
                                 if type(l) in [L.Conv2DLayer, L.DenseLayer]]
        if patterns is not None:
            if type(patterns) is list:
                patterns = patterns[0]
            for i,layer in enumerate(self.trainable_layers):
                pattern = patterns['A'][i]
                if pattern.ndim == 4:
                    pattern = pattern.transpose(1,0,2,3)
                elif pattern.ndim == 2:
                    pattern = pattern.T
                layer.W.set_value(pattern)

    def _put_rectifiers(self, input_layer, layer):
        return umisc.get_rectifier_copy_layer(input_layer, layer)

    def get_name(self):
        return 'PatternNet'


class GuidedPatternNetExplainer(PatternNetExplainer):

    def _put_rectifiers(self, input_layer, layer):
        tmp = umisc.get_rectifier_copy_layer(input_layer, layer)
        return umisc.get_rectifier_layer(tmp, layer)

    def get_name(self):
        return 'GuidedPatternNet'


class PatternLRPExplainer(PatternNetExplainer):

    def _set_inverse_parameters(self, patterns):
        self.trainable_layers = [(l, self.inverse_map[l])
                                 for l in L.get_all_layers(self.output_layer)
                                 if type(l) in [L.Conv2DLayer, L.DenseLayer]]
        if patterns is not None:
            if type(patterns) is list:
                patterns = patterns[0]
            for i,layer in enumerate(self.trainable_layers):
                param = layer[0].W.get_value()
                pattern = patterns['A'][i]
                if pattern.ndim == 4:
                    if layer[0].flip_filters:
                        param = param[:,:,::-1,::-1]
                    pattern = param*pattern
                    pattern = pattern.transpose(1,0,2,3)

                elif pattern.ndim == 2:
                    pattern = param*pattern
                    pattern = pattern.T
                layer[1].W.set_value(pattern)

    def show_as_rgb(self):
        return True
