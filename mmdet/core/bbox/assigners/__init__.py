# Copyright (c) OpenMMLab. All rights reserved.
from .approx_max_iou_assigner import ApproxMaxIoUAssigner
from .ascend_assign_result import AscendAssignResult
from .ascend_max_iou_assigner import AscendMaxIoUAssigner
from .assign_result import AssignResult
from .atss_assigner import ATSSAssigner
from .base_assigner import BaseAssigner
from .center_region_assigner import CenterRegionAssigner
from .grid_assigner import GridAssigner
from .hungarian_assigner import HungarianAssigner
from .mask_hungarian_assigner import MaskHungarianAssigner
from .max_iou_assigner import MaxIoUAssigner
from .point_assigner import PointAssigner
from .region_assigner import RegionAssigner
from .sim_ota_assigner import SimOTAAssigner
from .task_aligned_assigner import TaskAlignedAssigner
from .uniform_assigner import UniformAssigner
from .sim_ota_assigner_cyber import SimOTAAssigner_cyber
# from .sim_ota_assigner_lmy import SimOTAAssignerV2
# from .sim_ota_assigner01 import SimOTAAssignerV2
# from .sim_ota_assigner02 import SimOTAAssignerV2
# from .sim_ota_assigner03 import SimOTAAssignerV2
# from .sim_ota_assigner04 import SimOTAAssignerV2
# from .sim_ota_assigner05 import SimOTAAssignerV2
# from .sim_ota_assigner06 import SimOTAAssignerV2
# from .sim_ota_assigner07 import SimOTAAssignerV2
# from .sim_ota_assigner08 import SimOTAAssignerV2
# from .sim_ota_assigner09 import SimOTAAssignerV2
# from .sim_ota_assigner10 import SimOTAAssignerV2
# from .sim_ota_assigner11 import SimOTAAssignerV2
# from .sim_ota_assigner12 import SimOTAAssignerV2
# from .sim_ota_assigner13 import SimOTAAssignerV2
# from .sim_ota_assigner14 import SimOTAAssignerV2
# from .sim_ota_assigner15 import SimOTAAssignerV2
# from .sim_ota_assigner16 import SimOTAAssignerV2
# from .sim_ota_assigner17 import SimOTAAssignerV2
# from .sim_ota_assigner18 import SimOTAAssignerV2
# from .sim_ota_assigner19 import SimOTAAssignerV2
# from .sim_ota_assigner20 import SimOTAAssignerV2
# from .sim_ota_assigner21 import SimOTAAssignerV2
# from .sim_ota_assigner22 import SimOTAAssignerV2
# from .sim_ota_assigner23 import SimOTAAssignerV2
# from .sim_ota_assigner24 import SimOTAAssignerV2
from .sim_ota_assigner25 import SimOTAAssignerV2
# from .sim_ota_assigner26 import SimOTAAssignerV2
# from .sim_ota_assigner27 import SimOTAAssignerV2
# from .sim_ota_assigner28 import SimOTAAssignerV2
# from .sim_ota_assigner29 import SimOTAAssignerV2
# from .sim_ota_assigner30 import SimOTAAssignerV2 #25+giou
# from .sim_ota_assigner32 import SimOTAAssignerV2
# from .sim_ota_assigner33 import SimOTAAssignerV2
# from .sim_ota_assigner34 import SimOTAAssignerV2
# from .sim_ota_assigner35 import SimOTAAssignerV2
# from .sim_ota_assigner36 import SimOTAAssignerV2
# from .sim_ota_assigner37 import SimOTAAssignerV2
# from .sim_ota_assigner38 import SimOTAAssignerV2
# from .sim_ota_assigner39 import SimOTAAssignerV2
# from .sim_ota_assigner40 import SimOTAAssignerV2
# from .sim_ota_assigner41 import SimOTAAssignerV2
# from .sim_ota_assigner42 import SimOTAAssignerV2
# from .sim_ota_assigner43 import SimOTAAssignerV2
# from .sim_ota_assigner44 import SimOTAAssignerV2
# from .sim_ota_assigner45 import SimOTAAssignerV2
# from .sim_ota_assigner46 import SimOTAAssignerV2
# from .sim_ota_assigner47 import SimOTAAssignerV2
# from .sim_ota_assigner48 import SimOTAAssignerV2
# from .sim_ota_assigner49 import SimOTAAssignerV2

__all__ = [
    'BaseAssigner', 'MaxIoUAssigner', 'ApproxMaxIoUAssigner', 'AssignResult',
    'PointAssigner', 'ATSSAssigner', 'CenterRegionAssigner', 'GridAssigner',
    'HungarianAssigner', 'RegionAssigner', 'UniformAssigner', 'SimOTAAssigner',
    'TaskAlignedAssigner', 'MaskHungarianAssigner', 'AscendAssignResult',
    'AscendMaxIoUAssigner','SimOTAAssignerV2','SimOTAAssigner_cyber'
]
