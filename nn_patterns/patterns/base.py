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
import theano


__all__ = [
    "BasePatternComputer"
]


class BasePatternComputer(object):

    def __init__(self, output_layer):
        self.output_layer = output_layer
        self.layers = self._collect_layers()
        self.parameters = L.get_all_param_values(self.output_layer)
        self.stats_f = None

    def _get_split(self, layer, **kwargs):
        raise NotImplementedError("Has to be implemented by the subclass")

    def _process_batches(self, X_train, batch_size, **kwargs):
        raise NotImplementedError("Has to be implemented by the subclass")

    def _collect_layers(self):
        self.all_layers = L.get_all_layers(self.output_layer)
        ret = [l for l in self.all_layers if
                type(l) in [L.DenseLayer, L.Conv2DLayer]]
        
        return ret

    def _generate_pattern_expressions(self, layers, **kwargs):
        symbolic_patterns = []
        for layer in layers:
            symbolic_patterns.extend(list(self._get_split(layer, **kwargs)))
        return symbolic_patterns


    def compute_patterns(self, X_train, batch_size, n_batches=None, **kwargs):
        # 1) Collect the pattern theano expressions
        symbolic_patterns = self._generate_pattern_expressions(self.layers,
                                                               **kwargs)

        # 2) Compile the theano pattern component function
        xi = self.all_layers[0].input_var
        self.stats_f = theano.function(inputs=[xi], outputs=symbolic_patterns)

        # 3) Process the batches
        return self._process_batches(X_train, batch_size,
                                     n_batches=n_batches, **kwargs)
