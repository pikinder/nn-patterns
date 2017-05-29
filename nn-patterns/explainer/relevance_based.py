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

from .gradient_based import GradientExplainer
from .base import BaseRelevanceExplainer
from .base import BaseInvertExplainer
from ..utils import misc as umisc


__all__ = [
    "LRPZExplainer",
]


class LRPZExplainer(GradientExplainer):

    def explain(self, X, target=None, **kwargs):
        return X * super(LRPZExplainer, self).explain(X,
                                                      target=target,
                                                      **kwargs)

    def get_name(self):
        return "lrp-z"

    def show_as_rgb(self):
        return True
