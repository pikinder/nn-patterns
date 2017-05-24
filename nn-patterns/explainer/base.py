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


import lasagne.layers
import theano


__all__ = ["BaseExplainer"]


def __check_cpu_flaw__():
    # Todo: Check if float32 and float64 are still different on cpu.
    if theano.config.device[:3]=='cpu' and theano.config.floatX  != 'float64':
        raise RuntimeError("Results will be wrong "
                           "when running on cpu in float32")


class BaseExplainer(object):

    def __init__(self, output_layer, patterns=None, to_layer=None, **kwargs):
        __check_cpu_flaw__()

        self.original_output_layer = output_layer
        self.to_layer = to_layer

        layers = lasagne.layers.get_all_layers(output_layer)
        layers = layers[:self.to_layer]
        layers.append(lasagne.layers.FlattenLayer(layers[-1]))

        self.layers = layers
        self.output_layer = layers[-1]
        self.input_layer = layers[0]

        self._init_explain_function(patterns, **kwargs)
        pass

    def _init_explain_function(self, patterns,**kwargs):
        raise NotImplementedError("Has to be implemented by the subclass")

    def explain(self, X, target, **kwargs):
        raise NotImplementedError("Has to be implemented by the subclass")

    def get_name(self):
        raise NotImplementedError("Has to be implemented by the subclass")

    def show_as_rgb(self):
        return True
