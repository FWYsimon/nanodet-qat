import copy

from .gfl_head import GFLHead
from .nanodet_head import NanoDetHead
from .nanodet_plus_head import NanoDetPlusHead
from .simple_conv_head import SimpleConvHead
from .quantization_nanodet_plus_head import QuantizableNanoDetPlusHead

from .quantization_simple_conv_head import QuantizableSimpleConvHead

def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    if name == "GFLHead":
        return GFLHead(**head_cfg)
    elif name == "NanoDetHead":
        return NanoDetHead(**head_cfg)
    elif name == "NanoDetPlusHead":
        return NanoDetPlusHead(**head_cfg)
    elif name == "SimpleConvHead":
        return SimpleConvHead(**head_cfg)
    elif name == "QuantizableNanoDetPlusHead":
        return QuantizableNanoDetPlusHead(**head_cfg)
    elif name == "QuantizableSimpleConvHead":
        return QuantizableSimpleConvHead(**head_cfg)
    else:
        raise NotImplementedError
