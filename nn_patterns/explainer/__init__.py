
from .base import *

from .gradient_based import *
from .misc import *
from .pattern_based import *
from .relevance_based import *


def create_explainer(name,
                     output_layer, patterns=None, to_layer=None, **kwargs):
    return {
        # Gradient based
        "gradient": GradientExplainer,
        "gradient.alt": AlternativGradientExplainer,
        "deconvnet": DeConvNetExplainer,
        "guided": GuidedBackpropExplainer,

        # Relevance based
        "lrp.z": LRPZExplainer,
        "lrp.eps": LRPEpsExplainer,

        # Pattern based
        "patternnet": PatternNetExplainer,
        "patternnet.guided": PatternNetExplainer,
        "patternlrp": PatternLRPExplainer,
    }[name](output_layer, patterns=patterns, to_layer=to_layer, **kwargs)
