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


import numpy as np

from .base import BaseExplainer


__all__ = ["RandomExplainer", "InputExplainer"]


class InputExplainer(BaseExplainer):

    def explain(self, X, target=None, **kwargs):
        return X

    def _init_explain_function(self, patterns, **kwargs):
        pass

    def get_name(self):
        return "Input"

    def show_as_rgb(self):
        return True


class RandomExplainer(BaseExplainer):

    def explain(self, X, target=None, **kwargs):
        return np.random.randn(*X.shape)

    def _init_explain_function(self, patterns, **kwargs):
        pass

    def get_name(self):
        return "Random"

    def show_as_rgb(self):
        return True
