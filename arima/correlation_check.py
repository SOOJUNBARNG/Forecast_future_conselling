# データ操作ライブラリ
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 統計ライブラリ
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# 時間関連ライブラリ
from datetime import datetime

# UTILSから取るやつ

import sys
from pathlib import Path

# プロジェクトルートを `sys.path` に追加
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.print_output_in_matplotlib import plot_result
from utils.data_pre_process import data_process_group


# Get the current date
start_data = "2023-04-01"
start_data = pd.to_datetime(start_data)
current_date = pd.to_datetime(datetime.today().date())
current_date = pd.to_datetime("2025-03-28")
data_start_date = pd.to_datetime(f"{start_data}")

def arima_output():
    df = pd.read_csv("../data/counseling_count_group.csv")
    df = data_process_group(df, data_start_date, current_date)

    df = df[(df["date"] > start_data) & (df["date"] < current_date)]

        # データの基本統計量を表示
    print(df.describe())

    # 相関行列を表示
    corr_matrix = df.corr()

    return corr_matrix

print(arima_output())
print(arima_output().to_csv("arima_correlation.csv"))