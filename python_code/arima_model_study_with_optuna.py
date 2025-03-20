# データ操作ライブラリ
import itertools
import numpy as np
import pandas as pd

# 統計ライブラリ
import optuna
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load dataset
file_path = "../data/counseling_count_group.csv"
df = pd.read_csv(file_path)
df["日付"] = pd.to_datetime(df["日付"])
df = df[
    df["日付"] >= pd.Timestamp("2025-01-10") 
]
df = df.groupby("日付")["counseled"].sum().reset_index()
df.set_index("日付", inplace=True)


def arima_objective(trial):
    """Objective function for Optuna optimization."""
    p = trial.suggest_int("p", 0, 5)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 5)

    # Define exogenous variables
    exog_vars = ["feature1", "feature2"]  # Replace with actual column names
    exog = df[exog_vars] if exog_vars else None

    try:
        model = ARIMA(df["counseled"], order=(p, d, q), exog=exog,
                      enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit()
        return result.aic  # Minimize AIC
    except:
        return float("inf")  # Return high value if fitting fails

# P (Seasonal AutoRegressive order, SAR): Similar to p but for seasonal lags.
# D (Seasonal Differencing order, SI): Number of times seasonal differencing is applied.
# Q (Seasonal Moving Average order, SMA): Similar to q but for seasonal lags.
# S (Seasonal period): The length of the seasonal cycle (e.g., S=12 for monthly data, S=7 for weekly data).


def optuna_arima():
    # Optuna で最適化
    study = optuna.create_study(direction="minimize")
    study.optimize(arima_objective, n_trials=100)  # 50回試行

    # 最適なハイパーパラメータを取得
    best_params = study.best_params
    # print("Best ARIMA Parameters:", best_params)

    # 最適パラメータでモデルを再学習
    exog_vars = ["feature1", "feature2"]  # Use same exogenous variables
    exog = df[exog_vars] if exog_vars else None

    best_model = ARIMA(df["counseled"], order=(best_params["p"], best_params["d"], best_params["q"]),
                       exog=exog, enforce_stationarity=False, enforce_invertibility=False)
    
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

    # Define exogenous variables
    exog_vars = ["feature1", "feature2"]  # Replace with actual column names
    exog = df[exog_vars] if exog_vars else None

    try:
        # SARIMAX モデルの構築と学習
        model = SARIMAX(df["counseled"], 
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, S),
                        exog=exog,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        result = model.fit(disp=False)
        
        return result.aic  # AIC を最小化
    except:
        return np.inf  # エラー発生時は最悪値

def optuna_sarima():
    # Optuna で最適化
    study = optuna.create_study(direction="minimize")
    study.optimize(sarima_objective, n_trials=100)  # 50回試行

    # 最適なハイパーパラメータを取得
    best_params = study.best_params
    # print("Best SARIMA Parameters:", best_params)

    # Define exogenous variables
    exog_vars = ["feature1", "feature2"]  # Replace with actual column names
    exog = df[exog_vars] if exog_vars else None

    # 最適パラメータでモデルを再学習
    best_model = SARIMAX(df["counseled"],
                        order=(best_params["p"], best_params["d"], best_params["q"]),
                        seasonal_order=(best_params["P"], best_params["D"], best_params["Q"], 7),
                        exog=exog,
                        enforce_stationarity=False,
                        enforce_invertibility=False)

    best_result = best_model.fit(disp=False)
    # print("Best SARIMA AIC:", best_result.aic)

    return best_params



if __name__ == "__main__":
    print(optuna_arima())
    print(optuna_sarima())
