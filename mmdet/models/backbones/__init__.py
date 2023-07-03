# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
# from .csp_darknet_cyber import CSPDarknet_attention
# from .csp_darknet_cyber01 import CSPDarknet_attention
# from .csp_darknet_cyber02 import CSPDarknet_attention
# from .csp_darknet_cyber03 import CSPDarknet_attention
# from .csp_darknet_cyber04 import CSPDarknet_attention
# from .csp_darknet_cyber05 import CSPDarknet_attention
# from .csp_darknet_cyber06 import CSPDarknet_attention
# from .csp_darknet_cyber07 import CSPDarknet_attention
# from .csp_darknet_cyber08 import CSPDarknet_attention
# from .csp_darknet_cyber09 import CSPDarknet_attention
# from .csp_darknet_cyber10 import CSPDarknet_attention
from .csp_darknet_cyber11 import CSPDarknet_attention
# from .csp_darknet_cyber12 import CSPDarknet_attention


__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet','CSPDarknet_attention'
]
