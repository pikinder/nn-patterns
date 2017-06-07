
from .base import *

from .gradient_based import *
from .misc import *
from .pattern_based import *
from .relevance_based import *


def create_explainer(name,
                     output_layer, patterns=None, **kwargs):
    return {
        # Utility.
        "input": InputExplainer,
        "random": RandomExplainer,

        # Gradient based
        "gradient": GradientExplainer,
        "deconvnet": DeConvNetExplainer,
        "guided": GuidedBackpropExplainer,
        "gradient.alt": AlternativeGradientExplainer,

        # Relevance based
        "lrp.z": LRPZExplainer,
        "lrp.eps": LRPEpsExplainer,

        # Pattern based
        "patternnet": PatternNetExplainer,
        "patternnet.guided": GuidedPatternNetExplainer,
        "patternlrp": PatternLRPExplainer,
    }[name](output_layer, patterns=patterns, **kwargs)
