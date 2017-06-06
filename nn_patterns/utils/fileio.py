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


def save_parameters(filename, l):
    f = np.savez(filename, *l)
    pass


def load_parameters(filename):
    f = np.load(filename)
    ret = [f["arr_%i" % i] for i in range(len(f.keys()))]
    return ret


def store_patterns(filename, p):
    d = {}
    for prefix in ["A", "r", "mu"]:
        if prefix in p:
            d.update({"%s_%i" % (prefix, i): x
                      for i, x in enumerate(p[prefix])})
    np.savez(filename, **d)
    pass


def load_patterns(filename):
    f = np.load(filename)
    ret = {}
    for prefix in ["A", "r", "mu"]:
        l = sum([x.startswith(prefix) for x in f.keys()])
        ret.update({prefix: [f["%s_%i" % (prefix, i)] for i in range(l)]})
    return ret
