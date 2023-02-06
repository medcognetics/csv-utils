#!/usr/bin/env python
# -*- coding: utf-8 -*-

from csv_utils.input import concat, join


def test_join(df_factory):
    df1 = df_factory().loc[:, ("col1",)]
    df2 = df_factory().loc[:, ("col2",)]
    df3 = df_factory().loc[:, ("col3",)]
    df = join([df1, df2, df3])
    assert df.shape == (df1.shape[0], df1.shape[1] + df2.shape[1] + df3.shape[1])


def test_concat(df_factory):
    df1 = df_factory().iloc[:2]
    df2 = df_factory().iloc[2:4]
    df3 = df_factory().iloc[4:]
    df = concat([df1, df2, df3])
    assert (df == df_factory()).all().all()
