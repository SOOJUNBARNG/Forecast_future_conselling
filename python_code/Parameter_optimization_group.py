# データ操作ライブラリ
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

# 統計ライブラリ
import optuna
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

start_data = "2024-01-10"
start_data = pd.to_datetime(start_data)

# Load dataset
file_path = "../data/counseling_count_group.csv"
df = pd.read_csv(file_path)
df["日付"] = pd.to_datetime(df["日付"])
df = df[df["日付"] >= pd.Timestamp(f"{start_data}")]
df = df.groupby("日付")["counseled"].sum().reset_index()
df.set_index("日付", inplace=True)

current_date = pd.to_datetime(datetime.today().date())

import sys
from pathlib import Path

# プロジェクトルートを `sys.path` に追加
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.data_pre_process import data_process_group

def get_nth_week_of_month(date):
    first_day_of_month = date.replace(day=1)
    first_weekday = first_day_of_month.weekday()  # Monday = 0, Sunday = 6
    current_weekday = date.weekday()
    
    # Calculate the nth week of the month
    nth_week = (date.day + first_weekday) // 7 + 1
    return nth_week


# Get the current date
start_data = "2023-04-01"
start_data = pd.to_datetime(start_data)
current_date = pd.to_datetime(datetime.today().date())
current_date = pd.to_datetime("2025-03-28")
data_start_date = pd.to_datetime(f"{start_data}")

def data_process():
    df = pd.read_csv("../data/counseling_count_group.csv")
    df = data_process_group(df, data_start_date, current_date)
    df.to_csv("hello.csv", index=False)
    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    return df


def arima_objective(trial):
    """Objective function for Optuna optimization."""
    # {'p': 9, 'd': 1, 'q': 10}
    p = trial.suggest_int("p", 0, 10)
    d = trial.suggest_int("d", 0, 5)
    q = trial.suggest_int("q", 0, 10)

    data_process_df = pd.DataFrame(data_process())
    print(data_process_df.columns)
    data_process_df["date"] = pd.to_datetime(data_process_df["date"], errors="coerce")
    data_process_df.reset_index(inplace=True)
    data_process_df.dropna(subset=["date"], inplace=True)
    data_process_df.set_index("date", inplace=True)

    # Convert date column to datetime
    current_date = pd.to_datetime(datetime.today().date())
    df_exog = data_process_df.copy()

    # Define target variable
    y = data_process_df.loc[data_process_df.index <= current_date, "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[
        df_exog.index <= current_date,
        ["day", "counseled_lag1", "counseled_lag7", "counseled_lag28"],
    ]

    try:
        model = ARIMA(
            y,
            order=(p, d, q),
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit()
        return result.aic  # Minimize AIC
    except:
        return float("inf")  # Return high value if fitting fails


def optuna_arima():
    # Optuna で最適化
    study = optuna.create_study(direction="minimize")
    study.optimize(arima_objective, n_trials=30)  # 50回試行

    # 最適なハイパーパラメータを取得
    best_params = study.best_params
    # print("Best ARIMA Parameters:", best_params)

    data_process_df = pd.DataFrame(data_process())
    print(data_process_df.columns)
    data_process_df["date"] = pd.to_datetime(data_process_df["date"], errors="coerce")
    data_process_df.reset_index(inplace=True)
    data_process_df.dropna(subset=["date"], inplace=True)
    data_process_df.set_index("date", inplace=True)

    # Convert date column to datetime
    current_date = pd.to_datetime(datetime.today().date())
    df_exog = data_process_df.copy()

    # Define target variable
    y = data_process_df.loc[data_process_df.index <= current_date, "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[
        df_exog.index <= current_date,
        ["day", "counseled_lag1", "counseled_lag7", "counseled_lag28"],
    ]

    best_model = ARIMA(
        y,
        order=(best_params["p"], best_params["d"], best_params["q"]),
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    best_result = best_model.fit()
    # print("Best ARIMA AIC:", best_result.aic)

    return best_params


# 目的関数の定義
def sarima_objective(trial):
    # Non-seasonal parameters
    # {'p': 5, 'd': 1, 'q': 2}.
    p = trial.suggest_int("p", 0, 5)  # Increased range
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 5)  # Increased range

    # Seasonal parameters
    P = trial.suggest_int("P", 0, 5)  # Increased range
    D = trial.suggest_int("D", 0, 2)
    Q = trial.suggest_int("Q", 0, 5)  # Increased range
    # S = trial.suggest_int("S", 0, 100)  # Weekly, Monthly, Yearly
    S = trial.suggest_categorical("S", [7])  # Weekly or monthly seasonality

    data_process_df = pd.DataFrame(data_process())
    print(data_process_df.columns)
    data_process_df["date"] = pd.to_datetime(data_process_df["date"], errors="coerce")
    data_process_df.reset_index(inplace=True)
    data_process_df.dropna(subset=["date"], inplace=True)
    data_process_df.set_index("date", inplace=True)

    # Convert date column to datetime
    current_date = pd.to_datetime(datetime.today().date())
    df_exog = data_process_df.copy()

    # Define target variable
    y = data_process_df.loc[data_process_df.index <= current_date, "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[
        df_exog.index <= current_date,
        ["national_holiday", "clinic_holiday", "day_of_week"],
    ]

    try:
        # SARIMAX モデルの構築と学習
        model = SARIMAX(
            y,
            order=(p, d, q),
            seasonal_order=(P, D, Q, S),
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)

        return result.aic  # AIC を最小化
    except:
        return np.inf  # エラー発生時は最悪値


def optuna_sarima():
    # Optuna で最適化
    study = optuna.create_study(direction="minimize")
    study.optimize(sarima_objective, n_trials=300)  # 50回試行

    # 最適なハイパーパラメータを取得
    best_params = study.best_params
    # print("Best SARIMA Parameters:", best_params)

    data_process_df = pd.DataFrame(data_process())
    print(data_process_df.columns)
    data_process_df["date"] = pd.to_datetime(data_process_df["date"], errors="coerce")
    data_process_df.reset_index(inplace=True)
    data_process_df.dropna(subset=["date"], inplace=True)
    data_process_df.set_index("date", inplace=True)

    # Convert date column to datetime
    current_date = pd.to_datetime(datetime.today().date())
    df_exog = data_process_df.copy()

    # Define target variable
    y = data_process_df.loc[data_process_df.index <= current_date, "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[
        df_exog.index <= current_date,
        ["day", "counseled_lag1", "counseled_lag7", "counseled_lag28"],
    ]

    # 最適パラメータでモデルを再学習
    best_model = SARIMAX(
        y,
        order=(best_params["p"], best_params["d"], best_params["q"]),
        seasonal_order=(best_params["P"], best_params["D"], best_params["Q"], 7),
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    best_result = best_model.fit(disp=False)
    # print("Best SARIMA AIC:", best_result.aic)

    return best_params


if __name__ == "__main__":
    print(optuna_arima())
    # print(optuna_sarima())
