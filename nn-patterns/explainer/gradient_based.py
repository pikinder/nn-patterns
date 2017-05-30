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

from .base import BaseRelevanceExplainer
from .base import BaseInvertExplainer
from ..utils import misc as umisc


__all__ = [
    "GradientExplainer",
    "BaseDeConvNetExplainer",
    "DeConvNetExplainer",
    "GuidedBackpropExplainer",
    "AlternateGradientExplainer",
]


class GradientExplainer(BaseRelevanceExplainer):
    """
    Explainer that uses automatic differentiation
    to get the gradient of the input with respect to the output.
    """

    def _init_explain_function(self, patterns=None, **kwargs):
        with umisc.ignore_sigmoids(self.output_layer) as output_layer:
            Y = L.get_output(output_layer, deterministic=True)
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
                        explanation[i:i+1] += (self.grad_function(X[i:i+1],
                                                                  0, j) *
                                               relevance_values[i, j])
        return explanation

    def get_name(self):
        return "Gradient"


class BaseDeConvNetExplainer(BaseInvertExplainer):

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

    def _put_rectifiers(self,input_layer,layer):
        raise RuntimeError("Needs to be implemented by the subclass.")


class DeConvNetExplainer(BaseDeConvNetExplainer):

    def _invert_LocalResponseNormalisation2DLayer(self, layer, feeder):
        return feeder

    def _put_rectifiers(self, input_layer, layer):
        return umisc.get_rectifier_layer(input_layer, layer)

    def get_name(self):
        return "DeConvNet"


class GuidedBackpropExplainer(BaseDeConvNetExplainer):

    def _invert_LocalResponseNormalisation2DLayer(self, layer, feeder):
        return feeder

    def _put_rectifiers(self, input_layer, layer):
        input_layer = umisc.get_rectifier_layer(input_layer, layer)
        return umisc.get_rectifier_copy_layer(input_layer, layer)

    def get_name(self):
        return "Guided BackProp"


class AlternateGradientExplainer(BaseDeConvNetExplainer):

    def _invert_LocalResponseNormalisation2DLayer(self, layer, feeder):
        return L.InverseLayer(feeder,layer)

    def _put_rectifiers(self, input_layer, layer):
        return umisc.get_rectifier_copy_layer(input_layer, layer)

    def get_name(self):
        return "Gradient.alt"
