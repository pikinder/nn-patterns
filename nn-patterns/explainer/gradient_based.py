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
import theano.tensor as T

from .base import BaseRelevanceExplainer


__all__ = ["GradientExplainer"]


class GradientExplainer(BaseRelevanceExplainer):
    """
    Explainer that uses automatic differentiation
    to get the gradient of the input with respect to the output.
    """

    def _init_explain_function(self, patterns, **kwargs):
        output_nonlinearity = None
        if(hasattr(self.output_layer, 'nonlinearity') and
           (self.output_layer.nonlinearity == softmax or
            self.clf.output_layer.nonlinearity == sigmoid)):
            print("Removing the softmax or sigmoid output nonlinearity."
                  "for the explanation.")
            output_nonlinearity = self.output_layer.nonlinearity
            self.output_layer.nonlinearity = lambda x: x
        Y = lasagne.layers.get_output(self.output_layer, deterministic=True)
        if output_nonlinearity is not None:
            self.output_layer.nonlinearity = output_nonlinearity

        X = self.input_layer.input_var  # original
        I = T.iscalar()  # Output neuron
        S = T.iscalar()  # Sample that is desired
        E = T.grad(Y[S].flatten()[I], X)
        self.grad_function = theano.function(inputs=[X, S, I], outputs=E)

    def explain(self, X, target=None, **kwargs):
        explanation = np.zeros_like(X)
        relevance_values = self._get_relevance_values(X, target)
        for i in range(relevance_values.shape[0]):
            for j in range(relevance_values.shape[1]):
                    if relevance_values[i, j] != 0:
                        # Todo: Why do we do for each sample?
                        explanation[i:i+1] += (self.grad_function(X[i:i+1], 0, j) *
                                               relevance_values[i, j])
        return explanation

    def get_name(self):
        return "Gradient"
