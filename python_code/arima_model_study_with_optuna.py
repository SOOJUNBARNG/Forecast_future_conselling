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

# 目的関数の定義
def objective(trial):
    # p, d, q を選択
    p = trial.suggest_int("p", 0, 3)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 3)

    # P, D, Q, S (S=7 は週周期)
    P = trial.suggest_int("P", 0, 3)
    D = trial.suggest_int("D", 0, 2)
    Q = trial.suggest_int("Q", 0, 3)
    S = 7  # 週単位の周期性を考慮

    try:
        # SARIMAX モデルの構築と学習
        model = SARIMAX(df["counseled"], 
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, S),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        result = model.fit(disp=False)
        
        return result.aic  # AIC を最小化
    except:
        return np.inf  # エラー発生時は最悪値

# Optuna で最適化
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)  # 50回試行

# 最適なハイパーパラメータを取得
best_params = study.best_params
print("Best Parameters:", best_params)

# 最適パラメータでモデルを再学習
best_model = SARIMAX(df["counseled"],
                     order=(best_params["p"], best_params["d"], best_params["q"]),
                     seasonal_order=(best_params["P"], best_params["D"], best_params["Q"], 7),
                     enforce_stationarity=False,
                     enforce_invertibility=False)

best_result = best_model.fit(disp=False)
print("Best AIC:", best_result.aic)
