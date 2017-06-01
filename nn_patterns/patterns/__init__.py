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


from .base import BasePatternComputer
from .combined import CombinedPatternComputer


def compute_patterns(output_layer,
                     X_train,
                     batch_size,
                     n_batches=None,
                     **kwargs):
    pattern_computer = CombinedPatternComputer(output_layer)
    return pattern_computer.compute_patterns(X_train, batch_size,
                                             n_batches=n_batches, **kwargs)
