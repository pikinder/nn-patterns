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
import numpy as np
import theano
import theano.tensor as T

from .base import BasePatternComputer
from . import utils as putils


__all__ = [
    "CombinedPatternComputer"
]

# How everything fits together :
# Pattern types and the filter function.
subtypes = [
    ('basic', lambda x: 1.0+0.0*x),
    ('positive_y', lambda x: 1.0*T.gt(x,0.0)),
    ('negative_y', lambda x: 1.0-1.0*T.gt(x,0.0))
]
# Statistics need for each pattern type.
subtype_keys = [
    'cnt', # Number of sample per variable.
    'm_x', # Mean along x.
    'm_y', # Mean along y.
    'xty', # Covariance x and y.
    'yty', # Covaraince y and y.
]
# This has a specific aggregation function.
subtype_keys_no_aggregation = ['cnt']


# Create new stats dict.
def create_dict(new_stats):
    ret = []
    n_per_dict = len(subtypes)*len(subtype_keys)
    for i in range(0, len(new_stats), n_per_dict):
        ret.append(list_to_dict(new_stats[i : i+n_per_dict]))
    return ret

# Stats list to dict.
def list_to_dict(stats_list):
    stats_dict = dict()
    idx = 0
    for key, _ in subtypes:
        stats_dict[key]=dict()
        for sub_key in subtype_keys:
            stats_dict[key][sub_key] = stats_list[idx]
            idx+=1
    return stats_dict

# Stats dict to list
def dict_to_list(stats_dict):
    stats_list = []
    for key,_ in subtypes:
        for sub_key in subtype_keys:
            stats_list.append(stats_dict[key][sub_key])
    return stats_list


class CombinedPatternComputer(BasePatternComputer):

    def _get_split(self, layer,
                   deterministic=True, conv_all_patches=True, **kwargs):

        # Get the patches and the outputs without the non-linearities.
        if type(layer) is L.DenseLayer:
            x, y = putils.get_dense_xy(layer, deterministic)
        elif type(layer) is L.Conv2DLayer:
            if conv_all_patches is True:
                x, y = putils.get_conv_xy_all(layer, deterministic)
            else:
                x, y = putils.get_conv_xy(layer, deterministic)
        else:
            raise ValueError("Unknown layer as input")

        # Create an output dictionary
        outputs = dict()

        for name, fun in subtypes:
            outputs[name] = dict()
            mrk_y = 1.0* T.cast(fun(y), dtype=theano.config.floatX)  # (N,O)
            y_current = y*mrk_y # This has a binary mask
            cnt_y = T.shape_padaxis(T.sum(mrk_y, axis=0), axis=0)  # (1,O)
            norm = T.maximum(cnt_y, 1.)

            # Count how many datapoints are considered
            outputs[name]['cnt'] = cnt_y

            # The mean of the current batch
            outputs[name]['m_y'] = T.shape_padaxis(y_current.sum(axis=0), axis=0) / norm  # (1,O) mean output for batch
            outputs[name]['m_x'] = T.dot(x.T, mrk_y) / norm  # (D,O) mean input for batch

            # The mean of the current batch
            outputs[name]['yty'] = T.shape_padaxis(T.sum(y_current ** 2., axis=0), axis=0) / norm  # (1,O)
            outputs[name]['xty'] = T.dot(x.T, y_current) / norm  # D,O

        return dict_to_list(outputs)

    def _update_statistics(self, new_stats, stats):
        new_stats = create_dict(new_stats)
        if stats is None:
            stats = new_stats
            return stats

        # update the stats layerwise
        for l_i in range(len(stats)):

            for subtype,_ in subtypes:
                # TODO: Have to check the type to see if this is needed
                cnt_old = 1.0 * stats[l_i][subtype]['cnt']
                stats[l_i][subtype]['cnt'] = (stats[l_i][subtype]['cnt']
                                              + new_stats[l_i][subtype]['cnt'])
                norm = np.maximum(stats[l_i][subtype]['cnt'], 1.0)

                for key in subtype_keys:
                    if key not in subtype_keys_no_aggregation:
                        tmp_old = cnt_old / norm * stats[l_i][subtype][key]
                        tmp_new = (new_stats[l_i][subtype]['cnt']
                                   / norm * new_stats[l_i][subtype][key])
                        stats[l_i][subtype][key] = tmp_old + tmp_new
        return stats

    def _compute_Exy_ExEy(self,stats,key,l_i):
        return (stats[l_i][key]['xty']
                - stats[l_i][key]['m_x'] * stats[l_i]['basic']['m_y'])  # D,O

    def _get_W(self, id):
        dl = self.layers[id]
        W = dl.W.get_value()
        if W.ndim == 4:
            if dl.flip_filters:
                W = W[:, :, ::-1, ::-1]
            W = putils.flatten(W)
        return W

    def _update_length(self, A, id):
        W = self._get_W(id)
        norm = np.diag(np.dot(putils.flatten(W).T,A))[np.newaxis]
        norm =  norm + 1.0*(norm == 0.0)
        return A / norm

    def _compute_A(self, stats, key, l_i):
        W = self._get_W(l_i) #D,O
        numerator = self._compute_Exy_ExEy(stats, key, l_i) #D,O
        denumerator = np.dot(W.T,numerator) #O,O
        denumerator = np.diag(denumerator) #1,O
        if np.sum(denumerator == 0) > 0:
            denumerator= denumerator + 1.0*(denumerator==0)
        A = numerator / denumerator[np.newaxis]
        A = self._update_length(A, l_i)
        return A

    def _compute_patterns(self, stats):
        patterns = dict()
        for key,_ in subtypes:
            patterns[key]=dict()
            patterns[key]['A'] = []
            patterns[key]['r'] = []
            patterns[key]['mu'] = []

            for l_i in range(len(stats)):
                # using uppercase now
                A = self._compute_A(stats, key, l_i)
                r = stats[l_i][key]['m_x'] - A * stats[l_i][key]['m_y']  # D,O
                mu = stats[l_i][key]['m_x']

                if self.layers[l_i].W.get_value().ndim == 4:
                    A = A.T.reshape(self.layers[l_i].W.get_value().shape)
                    r = r.T.reshape(A.shape)
                    mu = mu.T.reshape(A.shape)

                assert(np.sum(np.isnan(A)) == 0.,
                       "Something went wrong, nan in A")

                patterns[key]['A'].append(A.astype(np.float32))
                patterns[key]['r'].append(r.astype(np.float32))
                patterns[key]['mu'].append(mu.astype(np.float32))
        return patterns

    def _process_batches(self, X_train, batch_size, n_batches=None, **kwargs):
        is_generator = type(X_train) not in [np.ndarray, np.core.memmap]

        if is_generator is True:
            if n_batches is None:
                raise ValueError("X_train is generator, in this case "
                                 "n_batches needs to be specified.")
        else:
            n_datapoints = X_train.shape[0]
            n_batches = n_datapoints // batch_size

        stats = None
        for i in range(n_batches):
            # Load batch
            if is_generator:
                X = X_train()
            else:
                X = X_train[i*batch_size : (i+1)*batch_size]

            # Get components
            new_stats = self.stats_f(X)
            # Update stats.
            stats= self._update_statistics(new_stats, stats)

        # Compute the actual patterns
        return self._compute_patterns(stats)
