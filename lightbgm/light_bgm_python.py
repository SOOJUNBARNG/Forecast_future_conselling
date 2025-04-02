# データ操作ライブラリ
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

# 統計・評価ライブラリ
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 時間関連ライブラリ
from datetime import datetime, timedelta

# UTILSから取るやつ
import sys
from pathlib import Path

# プロジェクトルートを `sys.path` に追加
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.print_output_in_matplotlib import plot_result
from utils.data_pre_process import data_process_group

# Get the current date
start_data = "2024-01-10"
start_data = pd.to_datetime(start_data)
current_date = pd.to_datetime(datetime.today().date())
current_date = pd.to_datetime("2025-03-01")
data_start_date = pd.to_datetime(f"{start_data}")


def objective(trial):
    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    
    df = pd.read_csv("../data/counseling_count_group.csv")
    df = data_process_group(df, data_start_date, current_date)
    df = prepare_features(df)
    
    train_df = df.loc[df.index <= pd.to_datetime(current_date)]
    target_col = "counseled"
    feature_cols = [col for col in train_df.columns if col != target_col]
    X_train, X_val, y_train, y_val = train_test_split(
        train_df[feature_cols], train_df[target_col], test_size=0.2, shuffle=False
    )
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params, train_data, valid_sets=[val_data]
        # ,verbose_eval=False
        # ,early_stopping_rounds=50
    )
    
    y_pred = model.predict(X_val)
    return mean_absolute_error(y_val, y_pred)

def retrun_best_value():

    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    print("Best parameters:", best_params)
    return best_params

def prepare_features(df):
    """特徴量エンジニアリング"""
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # ラグ特徴量（過去の値を利用）
    for lag in range(1, 8):
        df[f"lag_{lag}"] = df["counseled"].shift(lag)

    # 移動平均特徴量
    df["rolling_mean_7"] = df["counseled"].shift(1).rolling(window=7).mean()

    # ダミー変数化（曜日）
    df["weekday"] = df.index.weekday
    df = pd.get_dummies(df, columns=["weekday"], drop_first=True)

    # 休日フラグ
    df["national_holiday"] = df["national_holiday"].astype(int)
    df["clinic_holiday"] = df["clinic_holiday"].astype(int)

    # 欠損値を補完
    df.fillna(0, inplace=True)
    
    return df

def lightgbm_forecast():
    df = pd.read_csv("../data/counseling_count_group.csv")
    df = data_process_group(df, data_start_date, current_date)
    df = prepare_features(df)

    # 学習データの範囲を決定
    train_df = df.loc[df.index <= pd.to_datetime(current_date)]
    test_df = df.loc[df.index > pd.to_datetime(current_date)][:28]

    # 目的変数と説明変数
    target_col = "counseled"
    feature_cols = [col for col in train_df.columns if col != target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        train_df[feature_cols], train_df[target_col], test_size=0.2, shuffle=False
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = retrun_best_value()
    # LightGBMモデルのパラメータ
    # params = {
    #     "objective": "regression",
    #     "metric": "mae",
    #     "boosting_type": "gbdt",
    #     "learning_rate": 0.05,
    #     "num_leaves": 31,
    #     "max_depth": -1,
    #     "feature_fraction": 0.8,
    #     "bagging_fraction": 0.8,
    #     "bagging_freq": 5,
    #     "verbose": -1,
    # }

    # 学習
    model = lgb.train(
        params, 
        train_data, 
        valid_sets=[train_data, val_data], 
        valid_names=["train", "valid"],  # 追加
        num_boost_round=500, 
        # early_stopping_rounds=50
        )

    # 予測
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # エラーメトリクス
    mae = mean_absolute_error(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # 未来の28日間を予測
    forecast = model.predict(test_df[feature_cols])

    # 予測結果をデータフレームにまとめる
    forecast_index = pd.date_range(
        start=train_df.index[-1] + timedelta(days=1), periods=28, freq="D"
    )

    forecast_df = pd.DataFrame(
        {"Date": forecast_index, "Forecast": forecast.astype(int)}
    )

    # 予測結果を保存
    forecast_df.to_csv("../data/lightgbm_forecast_all_clinics.csv", index=False)

def main():
    lightgbm_forecast()

if __name__ == "__main__":
    main()
