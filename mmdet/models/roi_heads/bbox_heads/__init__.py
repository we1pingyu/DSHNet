from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .double_bbox_head import DoubleConvFCBBoxHead
from .transformer_bbox_head import TransformerBBoxHead
from .tail_bbox_head import TailBBoxHead
from .tail_gs_bbox_head import TailGSBBoxHead

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'TransformerBBoxHead', 'TailBBoxHead', 'TailGSBBoxHead'
]
