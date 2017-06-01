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


from ...utils.tests import dryrun

from ...explainer import PatternNetExplainer
from ...explainer import GuidedPatternNetExplainer
from ...explainer import PatternLRPExplainer


class TestPatterNetExplainer(dryrun.ExplainerTestCase):

    def _method(self, output_layer):
        return PatternNetExplainer(output_layer)

class TestGuidedPatterNetExplainer(dryrun.ExplainerTestCase):

    def _method(self, output_layer):
        return GuidedPatternNetExplainer(output_layer)

class TestPatterLRPExplainer(dryrun.ExplainerTestCase):

    def _method(self, output_layer):
        return PatternLRPExplainer(output_layer)
