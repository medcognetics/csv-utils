#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from .version import __version__
except ImportError:
    __version__ = "Unknown"

from .input import AGGREGATOR_REGISTRY, INPUT_REGISTRY
from .run import transform_csv
from .transforms import TRANSFORM_REGISTRY, Transform


__all__ = ["INPUT_REGISTRY", "AGGREGATOR_REGISTRY", "TRANSFORM_REGISTRY", "transform_csv", "Transform"]
