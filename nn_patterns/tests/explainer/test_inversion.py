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

from ...utils.tests import dryrun

from ...explainer import GradientExplainer
from ...explainer import AlternateGradientExplainer


class TestInversion(dryrun.TestCase):

    def _method(self, output_layer):
        return GradientExplainer(output_layer)

    def _assert(self, method, network, x, explanation):
        # Test if Theano gradient is close to the inverted gradient.
        # This tests that we inverted the network in the correct way.
        explainer_alt = AlternateGradientExplainer(network["out"])
        explanation_alt = explainer_alt.explain(x)

        np.testing.assert_allclose(explanation, explanation_alt)
        pass
