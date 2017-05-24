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
import lasagne.nonlinearities
import numpy as np
import theano


__all__ = ["BaseExplainer", "BaseRelevanceExplainer"]


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

    def explain(self, X, target=None, **kwargs):
        raise NotImplementedError("Has to be implemented by the subclass")

    def get_name(self):
        raise NotImplementedError("Has to be implemented by the subclass")

    def show_as_rgb(self):
        return True


class BaseRelevanceExplainer(BaseExplainer):

    def __init__(self, *args, **kwargs):
        super(BaseRelevanceExplainer, self).__init__(*args, **kwargs)
        self._init_relevance_function()
        pass

    def _init_relevance_function(self):
        if(hasattr(self.output_layer,'nonlinearity') and
           self.output_layer.nonlinearity == lasagne.nonlinearities.softmax):
            # Ignore softmax.
            output_nonlinearity = self.output_layer.nonlinearity
            self.output_layer.nonlinearity = lambda x:x
            output = lasagne.layers.get_output(self.output_layer,
                                               deterministic=True)
            self.output_layer.nonlinearity = output_nonlinearity
        else:
            output = lasagne.layers.get_output(self.output_layer,
                                               deterministic=True)

        self.relevance_function = theano.function(
            inputs=[self.input_layer.input_var], outputs=output)
        pass

    def _get_relevance_values(self, X, target):
        if target == 'max_output' or target == 'max_output_as_one':
            predictions = self.relevance_function(X)
            argmax = predictions.argmax(axis=1)
            relevance_values = np.zeros_like(predictions)

            for i in range(len(argmax)):
                relevance_values[i, argmax[i]] = 1.
                if target == 'max_output':
                    relevance_values[i, argmax[i]] *= predictions[i, argmax[i]]

        elif target is None:
            relevance_values = self.relevance_function(X)
        else:
            relevance_values = target

        return relevance_values.astype(X.dtype)
