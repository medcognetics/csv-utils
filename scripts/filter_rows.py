#!/usr/bin/env python
# -*- coding: utf-8 -*-

from csv_utils import TRANSFORM_REGISTRY, transform_csv
from csv_utils.transforms import KeepWhere, KeepColumns, DropWhere
from csv_utils.__main__ import main, parse_args
from functools import partial

DropMalign = DropWhere(column="Malignant", value="malignant")
TRANSFORM_REGISTRY(DropMalign, name="drop-malign")
DropBenign = DropWhere(column="Malignant", value="benign")
TRANSFORM_REGISTRY(DropBenign, name="drop-benign")

# to find postsurgical
KeepPostsurgical = DropWhere(column="Post Surgical", value="--")
TRANSFORM_REGISTRY(KeepPostsurgical, name="keep-ps")
KeepCols = KeepColumns(columns=["Post Surgical", "Study Path", "Case Notes"])
TRANSFORM_REGISTRY(KeepCols, name="keep-ps-cols")

# to find lesion size
KeepPostsurgical = KeepWhere(column="Lesion Size", value="???")
TRANSFORM_REGISTRY(KeepPostsurgical, name="keep-ls")
KeepCols = KeepColumns(columns=["Lesion Size", "# ROIs", "Study Path"])
TRANSFORM_REGISTRY(KeepCols, name="keep-ls-cols")

# to find pathology type
KeepPostsurgical = KeepWhere(column="Pathology Type", value="???")
TRANSFORM_REGISTRY(KeepPostsurgical, name="keep-pt")
KeepCols = KeepColumns(columns=["Study Path", "Pathology Type", "Case Notes"])
TRANSFORM_REGISTRY(KeepCols, name="keep-pt-cols")

# to find missing BIRADS 
KeepPostsurgical = KeepWhere(column="BIRADS", value="???")
TRANSFORM_REGISTRY(KeepPostsurgical, name="keep-br")
KeepCols = KeepColumns(columns=["BIRADS", "Malignant", "Study Path", "Case Notes"])
TRANSFORM_REGISTRY(KeepCols, name="keep-br-cols")

# to find lesion size
KeepPostsurgical = KeepWhere(column="Lesion Type", value="???")
TRANSFORM_REGISTRY(KeepPostsurgical, name="keep-lt")
KeepCols = KeepColumns(columns=["Lesion Type", "Study Path"])
TRANSFORM_REGISTRY(KeepCols, name="keep-lt-cols")


if __name__ == "__main__":
    main(parse_args())
