import optuna
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load dataset
file_path = "../data/counseling_count.csv"
df = pd.read_csv(file_path)
df["日付"] = pd.to_datetime(df["日付"])
df = df.groupby("日付")["counseled"].sum().reset_index()
df.set_index("日付", inplace=True)

# P (Seasonal AutoRegressive order, SAR): Similar to p but for seasonal lags.
# D (Seasonal Differencing order, SI): Number of times seasonal differencing is applied.
# Q (Seasonal Moving Average order, SMA): Similar to q but for seasonal lags.
# S (Seasonal period): The length of the seasonal cycle (e.g., S=12 for monthly data, S=7 for weekly data).


# 目的関数の定義
def objective(trial):
    # Non-seasonal parameters
    p = trial.suggest_int("p", 0, 5)  # Increased range
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 5)  # Increased range

    # Seasonal parameters
    P = trial.suggest_int("P", 0, 5)  # Increased range
    D = trial.suggest_int("D", 0, 2)
    Q = trial.suggest_int("Q", 0, 5)  # Increased range
    S = trial.suggest_categorical("S1", [7, 30, 365])  # Weekly, Monthly, Yearly
    # S = trial.suggest_categorical("S", [7, 12])  # Weekly or monthly seasonality

    try:
        # SARIMAX モデルの構築と学習
        model = SARIMAX(
            df["counseled"],
            order=(p, d, q),
            seasonal_order=(P, D, Q, S),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)

        return result.aic  # AIC を最小化
    except:
        return np.inf  # エラー発生時は最悪値


# Optuna で最適化
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)  # 50回試行

# 最適なハイパーパラメータを取得
best_params = study.best_params
print("Best Parameters:", best_params)

# 最適パラメータでモデルを再学習
best_model = SARIMAX(
    df["counseled"],
    order=(best_params["p"], best_params["d"], best_params["q"]),
    seasonal_order=(best_params["P"], best_params["D"], best_params["Q"], 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

best_result = best_model.fit(disp=False)
print("Best AIC:", best_result.aic)
