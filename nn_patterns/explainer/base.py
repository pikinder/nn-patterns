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


from ..utils import misc as umisc


__all__ = ["BaseExplainer", "BaseRelevanceExplainer"]


def __check_cpu_flaw__():
    # Todo: Check if float32 and float64 are still different on cpu.
    if theano.config.device[:3]=='cpu' and theano.config.floatX  != 'float64':
        raise RuntimeError("Results will be wrong "
                           "when running on cpu in float32")


class BaseExplainer(object):

    def __init__(self, output_layer, patterns=None, **kwargs):
        __check_cpu_flaw__()

        layers = L.get_all_layers(output_layer)
        self.layers = layers
        self.input_layer = layers[0]
        self.output_layer = layers[-1]

        self._init_explain_function(patterns=patterns, **kwargs)
        pass

    def _init_explain_function(self, patterns=None,**kwargs):
        raise NotImplementedError("Has to be implemented by the subclass")

    def explain(self, X, target="max_output", **kwargs):
        raise NotImplementedError("Has to be implemented by the subclass")

    def get_name(self):
        raise NotImplementedError("Has to be implemented by the subclass")

    def show_as_rgb(self):
        return True


class BaseRelevanceExplainer(BaseExplainer):

    def __init__(self, *args, **kwargs):
        super(BaseRelevanceExplainer, self).__init__(*args, **kwargs)
        self._init_relevance_function()
        pass

    def _init_relevance_function(self):
        with umisc.ignore_sigmoids(self.output_layer) as output_layer:
            output = L.get_output(output_layer,
                                  deterministic=True)

        self.relevance_function = theano.function(
            inputs=[self.input_layer.input_var], outputs=output)
        pass

    def _get_relevance_values(self, X, target="max_output"):
        if target == "max_output":
            predictions = self.relevance_function(X)
            argmax = predictions.argmax(axis=1)
            relevance_values = np.zeros_like(predictions)

            for i in range(len(argmax)):
                relevance_values[i, argmax[i]] = 1.
                if target == 'max_output':
                    relevance_values[i, argmax[i]] *= predictions[i, argmax[i]]

        elif target is None:
            relevance_values = self.relevance_function(X)
        else:
            relevance_values = target
        return relevance_values.astype(X.dtype)


class BaseInvertExplainer(BaseRelevanceExplainer):
    """
    Explainer model that uses layerwise inversion...
    """

    def _put_rectifiers(self, input_layer, layer):
        raise NotImplementedError("Subclass responsability")

    def _set_inverse_parameters(self, patterns=None):
        raise NotImplementedError("Subclass responsability")

    def _invert_UnknownLayer(self, layer, feeder):
        raise NotImplementedError("Subclass responsability. Layer type: %s" %
                                  type(layer))

    def _invert_DenseLayer(self, layer, feeder):
        # Warning they are swapped here
        feeder = self._put_rectifiers(feeder, layer)
        output_units = np.prod(L.get_output_shape(layer.input_layer)[1:])
        output_layer = L.DenseLayer(feeder,
                                    num_units=output_units,
                                    nonlinearity=None, b=None)
        return output_layer

    def _invert_FlattenLayer(self, layer, feeder):
        # The reshaping is handled in the main code.
        return feeder

    def _invert_Conv2DLayer(self, layer, feeder):
        def _check_padding_same():
            for s, p in zip(layer.filter_size, layer.pad):
                if s % 2 != 1:
                    return False
                elif s//2 != p:
                    return False
            return True

        # Warning they are swapped here.
        feeder = self._put_rectifiers(feeder,layer)

        f_s = layer.filter_size
        if layer.pad == 'same' or _check_padding_same():
            pad = 'same'
        elif layer.pad == 'valid' or layer.pad == (0, 0):
            pad = 'full'
        else:
            raise RuntimeError("Define your padding as full or same.")

        # By definition the
        # Flip filters must be on to be a proper deconvolution.

        num_filters = L.get_output_shape(layer.input_layer)[1]
        if layer.stride == (4,4):
            # Todo: clean this!
            print("Applying alexnet hack.")
            feeder = L.Upscale2DLayer(feeder, layer.stride, mode='dilate')
            output_layer = L.Conv2DLayer(feeder,
                                         num_filters=num_filters,
                                         filter_size=f_s,
                                         stride=1, pad=pad,
                                         nonlinearity=None, b=None,
                                         flip_filters=True)
            print("Applying alexnet hack part 2.")
            conv_layer = output_layer
            output_layer = L.SliceLayer(L.SliceLayer(output_layer,
                                                     slice(0,-3), axis=3),
                                        slice(0,-3), axis=2)
            output_layer.W = conv_layer.W
        elif layer.stride == (2,2):
            # Todo: clean this! Seems to be the same code as for AlexNet above.
            print("Applying GoogLeNet hack.")
            feeder = L.Upscale2DLayer(feeder, layer.stride, mode='dilate')
            output_layer = L.Conv2DLayer(feeder,
                                         num_filters=num_filters,
                                         filter_size=f_s,
                                         stride=1, pad=pad,
                                         nonlinearity=None, b=None,
                                         flip_filters=True)
        else:
            # Todo: clean this. Repetitions all over.
            output_layer = L.Conv2DLayer(feeder,
                                         num_filters=num_filters,
                                         filter_size=f_s,
                                         stride=1, pad=pad,
                                         nonlinearity=None, b=None,
                                         flip_filters=True)
        return output_layer

    def _invert_LocalResponseNormalisation2DLayer(self, layer, feeder):
        raise NotImplementedError("Subclass responsability")

    def _invert_DropoutLayer(self, layer, feeder):
        assert isinstance(layer, L.DropoutLayer)
        return feeder

    def _invert_InputLayer(self, layer, feeder):
        assert isinstance(layer, L.InputLayer)
        return feeder

    def _invert_GlobalPoolLayer(self, layer, feeder):
        assert isinstance(layer, L.GlobalPoolLayer)
        assert layer.pool_function == T.mean
        assert len(L.get_output_shape(layer.input_layer)) == 4

        target_shape = L.get_output_shape(feeder)+(1,1)
        if target_shape[0] is None:
            target_shape = (-1,) + target_shape[1:]

        feeder = L.ReshapeLayer(feeder, target_shape)

        upscaling = L.get_output_shape(layer.input_layer)[2:]
        feeder = L.Upscale2DLayer(feeder, upscaling)

        def expression(x):
            return x / np.prod(upscaling).astype(theano.config.floatX)
        feeder = L.ExpressionLayer(feeder, expression)
        return feeder

    def _invert_PadLayer(self, layer, feeder):
        assert isinstance(layer, L.PadLayer)
        assert layer.batch_ndim == 2
        assert len(L.get_output_shape(layer))==4.

        tmp = L.SliceLayer(feeder,
                           slice(layer.width[0][0], -layer.width[0][1]),
                           axis=2)
        return L.SliceLayer(tmp,
                            slice(layer.width[1][0], -layer.width[1][1]),
                            axis=3)

    def _invert_MaxPoolingLayer(self, layer, feeder):
        assert type(layer) in [L.MaxPool2DLayer, L.MaxPool1DLayer]
        return L.InverseLayer(feeder, layer)

    def _invert_SliceLayer(self, layer, feeder):
        # The undoing of the slicing is considered
        # in the _invert_layer_recursion function.
        return feeder

    def _invert_layer(self, layer, feeder):
        layer_type = type(layer)

        if L.get_output_shape(feeder) != L.get_output_shape(layer):
            feeder = L.ReshapeLayer(feeder, (-1,)+L.get_output_shape(layer)[1:])
        if layer_type is L.InputLayer:
            return self._invert_InputLayer(layer, feeder)
        elif layer_type is L.FlattenLayer:
            return self._invert_FlattenLayer(layer, feeder)
        elif layer_type is L.DenseLayer:
            return self._invert_DenseLayer(layer, feeder)
        elif layer_type is L.Conv2DLayer:
            return self._invert_Conv2DLayer(layer, feeder)
        elif layer_type is L.DropoutLayer:
            return self._invert_DropoutLayer(layer, feeder)
        elif layer_type in [L.MaxPool2DLayer, L.MaxPool1DLayer]:
            return self._invert_MaxPoolingLayer(layer, feeder)
        elif layer_type is L.PadLayer:
            return self._invert_PadLayer(layer, feeder)
        elif layer_type is L.SliceLayer:
            return self._invert_SliceLayer(layer, feeder)
        elif layer_type is L.LocalResponseNormalization2DLayer:
            return self._invert_LocalResponseNormalisation2DLayer(layer, feeder)
        elif layer_type is L.GlobalPoolLayer:
            return self._invert_GlobalPoolLayer(layer, feeder)
        else:
            return self._invert_UnknownLayer(layer, feeder)

    def _invert_layer_recursion(self, layer, prev_layer):
        """
        Note for concatenation layers this will be called multiple times.
        :param layer: Start the inversion recusrion in this layer.
        :return: the inverted layer, part of the entire graph.
        """
        # If we have a concatenation layer,
        # we must find out at which point it is concatenated and slice.
        # We should not store it in the map for this layer.
        # Because that would corrupt the result.

        # Did we already invert this?
        if self.inverse_map[layer] is not None:
            return self.inverse_map[layer]

        feeder = [self._invert_layer_recursion(l, layer)
                  for l in self.output_map[layer]]

        # Concatenation layers must be handled here.
        # This is not elegant, but it is important for the recursion that
        # the correct slice is computed every single time

        # Find the inverse of the layers this one feeds.
        # If this is none, it is the top layer and
        # we have to inject the explanation starting point
        if len(feeder) == 1:
            feeder = feeder[0]
        elif len(feeder) == 0:
            # It feeds nothing, so must be
            # output layer with restricted assumptions
            def nonlinearity(x):
                return 0 * x + self.relevance_values
            feeder = L.NonlinearityLayer(layer,
                                         nonlinearity=nonlinearity)
        else:
            # Multiple feeders.
            if type(self.output_map[layer][0]) is SliceLayer:
                print("Assuming all slices and non-overlapping")
                # TODO CHECK ASSUMPTIONS ARE APPLICABLE
                cat_axis = self.output_map[layer][0].axis
                print([l.slice for l in self.output_map[layer]])
                feeder = L.ConcatLayer(feeder, axis=cat_axis)
            else:
                feeders = feeder
                feeder = feeders[0]
                for f in feeders[1:]:
                    feeder = L.ElemwiseSumLayer([feeder, f])

        # Concatenation layer or other layer.
        if isinstance(layer, L.ConcatLayer):
            axis = layer.axis
            start_slice = 0
            for l in  layer.input_layers:
                if l == prev_layer:
                    break
                start_slice += L.get_output_shape(l)[axis]
            end_slice = start_slice + L.get_output_shape(prev_layer)[axis]
            return L.SliceLayer(feeder,
                                slice(start_slice, end_slice),
                                axis=axis)
        else:
            self.inverse_map[layer] = self._invert_layer(layer, feeder)
            return self.inverse_map[layer]

    def _construct_layer_maps(self):
        layers = L.get_all_layers(self.output_layer)
        # Store inverse layers to enable merging.
        self.inverse_map = {l: None for l in layers}
        # Store the layers a specific layer feeds.
        self.output_map = {l: [] for l in layers}

        for layer in  layers:
            if type(layer) is not L.InputLayer:
                if isinstance(layer, L.MergeLayer):
                    for feeder in layer.input_layers:
                        self.output_map[feeder].append(layer)
                else:
                    self.output_map[layer.input_layer].append(layer)

    def _remove_softmax(self):
        # Todo: should do this in a way that does not alter input.
        umisc.remove_sigmoids(self.output_layer)
                    
    def _init_network(self, patterns=None, **kwargs):
        self._remove_softmax()
        self.relevance_values = T.matrix()
        self._construct_layer_maps()
        tmp = self._invert_layer_recursion(self.input_layer, None)
        self.explain_output_layer = tmp

        # Call in any case. Patterns are not always needed.
        self._set_inverse_parameters(patterns=patterns)
        #print("\n\n\nNetwork")
        #for l in get_all_layers(self.explain_output_layer):
        #    print(type(l), get_output_shape(l))
        #print("\n\n\n")

    def _init_explain_function(self, patterns=None, **kwargs):
        self._init_network(patterns=patterns)
        explanation = L.get_output(self.explain_output_layer,
                                   deterministic=True)
        self.explain_function = theano.function(
            inputs=[self.input_layer.input_var, self.relevance_values],
            outputs=explanation)

    def explain(self, X, target="max_output", **kwargs):
        relevance_values = self._get_relevance_values(X, target)
        return self.explain_function(X, relevance_values.astype(X.dtype))
