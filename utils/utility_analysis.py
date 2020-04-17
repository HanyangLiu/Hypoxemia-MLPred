import os
import pandas as pd
import numpy as np
from file_config.config import config


def feature_value_filling_ratio(df):
    ratios = []
    for column in list(df.columns):
        a = df[df[column].isnull()]
        ratio = 1 - len(a) / len(df)
        ratios.append(ratio)

    filling_ratio = dict(zip(list(df.columns), ratios))

    return filling_ratio
