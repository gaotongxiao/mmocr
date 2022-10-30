# Copyright (c) OpenMMLab. All rights reserved.
from .base_preprocessor import BasePreprocessor
from .tps_preprocessor import STN, TPStransform

__all__ = ['BasePreprocessor', 'TPStransform', 'STN']
