# -*- coding: utf-8 -*-
"""init.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HtMpNODDbYcPWTBvFGzTXtsdgBgUiY0T
"""

#import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from scipy.stats import t
from scipy.optimize import minimize
from scipy.stats import kurtosis
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

from sklearn.decomposition import PCA

from scipy.stats import linregress

from datetime import datetime
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

!rm -rf FinTech545_Fall2024
!git clone <repository_url>

!git clone https://github.com/dompazz/FinTech545_Fall2024.git

!rm -rf Fintech545-Wenqi-Cai
!git clone <repository_url>

!git clone https://github.com/Wenqi-Cai/Fintech545-Wenqi-Cai.git

from cov import*
import PSD
import simulation
import returns
import regression
import Var
import option

#数据加载与预处理模块
def preprocess_data(data, drop_column=None, filter_column=None):

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    if drop_column and drop_column in data.columns:
        data = data.drop(columns=[drop_column])

    if filter_column and filter_column in data.columns:
        data = data[data[filter_column].notnull()]

    remaining_columns = list(data.columns)
    return data, remaining_columns

# processed_data, column_names = preprocess_data(data, drop_column="Column1", filter_column="SPY")

#数据类型转换将 DataFrame 中的字符串类型列转换为浮点数类型
def convert_columns_to_float(df):

    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"Converting column: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df