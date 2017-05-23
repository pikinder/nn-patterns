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


import theano.tensor as T

from . import mnist
from . import cifar10
from . import imagenet


def iterator():
    """
    Iterator over various networks.
    """

    default_nonlinearity = T.nnet.relu

    def fetch_networks(module_name, module):
        ret = [("%s.%s" % (module_name, name),
                getattr(module, name)(default_nonlinearity))
               for name in module.__all__]

        for name, network in ret:
            network["name"] = name

        return [x[1] for x in sorted(ret)]

    networks = (
        fetch_networks("mnist", mnist) +
        fetch_networks("cifar10", cifar10) +
        fetch_networks("imagenet", imagenet)
    )

    for network in networks:
        yield network
