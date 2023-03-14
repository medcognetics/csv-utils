#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata

from .input import AGGREGATOR_REGISTRY, INPUT_REGISTRY
from .run import transform_csv
from .transforms import TRANSFORM_REGISTRY, Transform


__version__ = importlib.metadata.version("csv-utils")


__all__ = ["INPUT_REGISTRY", "AGGREGATOR_REGISTRY", "TRANSFORM_REGISTRY", "transform_csv", "Transform"]
