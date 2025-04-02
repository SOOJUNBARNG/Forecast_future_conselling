import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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

# データの前処理
def prepare_lstm_data(df, look_back=14):  # Increased look_back period
    """LSTM用に時系列データを準備"""
    df = data_process_group(df, data_start_date, current_date)

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # 必要な列を取得
    target_col = "counseled"
    feature_cols = ["national_holiday", "clinic_holiday", "day", "week_of_month"]
    
    # ラグ特徴の追加（過去のデータを特徴として使う）
    for lag in range(1, look_back + 1):
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    
    # 移動平均の追加
    df["rolling_mean_7"] = df[target_col].rolling(window=7).mean()
    df["rolling_std_7"] = df[target_col].rolling(window=7).std()
    
    # 曜日ごとの平均を追加
    df["day_mean"] = df.groupby("day")[target_col].transform("mean")
    
    # 月の週の平均を追加
    df["week_of_month_mean"] = df.groupby("week_of_month")[target_col].transform("mean")

    # NaNを削除（最初の数行はラグの影響でNaNが入る）
    df.dropna(inplace=True)

    # 説明変数と目的変数を分割
    feature_cols.extend([
        "rolling_mean_7", "rolling_std_7", "day_mean", "week_of_month_mean",
        *[f"lag_{i}" for i in range(1, look_back + 1)]
    ])
    
    X = df[feature_cols].values
    y = df[target_col].values

    # データを0-1にスケーリング
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    print(f"Input shape: {X.shape}, Features: {len(feature_cols)}")

    # LSTMの入力形式に変換（[サンプル数, タイムステップ, 特徴量数]）
    global look_back_count
    look_back_count = look_back + len(feature_cols)
    X = X.reshape((X.shape[0], look_back_count, -1))

    return X, y, scaler, df.index

# LSTMモデルの構築
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, activation="relu", return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, activation="relu", return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(16, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model

# 予測処理
def lstm_forecast():
    df = pd.read_csv("../data/counseling_count_group.csv")
    # データの準備
    look_back = 14  # 過去14日間を使用
    X, y, scaler, date_index = prepare_lstm_data(df, look_back)

    # 学習データとテストデータに分割
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # LSTMモデルの構築
    model = build_lstm_model((look_back, X.shape[2]))

    # Early stoppingとModel checkpointの設定
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        "best_lstm_model.h5",
        monitor="val_loss",
        save_best_only=True
    )

    # 学習
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # 予測
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 精度評価
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # 未来の28日間を予測
    forecast = []
    last_data = X_test[-1].reshape(1, look_back_count, -1)  # 直近のデータ

    for _ in range(28):
        next_pred = model.predict(last_data)[0, 0]  # 予測値
        forecast.append(next_pred)
        last_data = np.roll(last_data, shift=-1, axis=1)  # 古いデータを削除
        last_data[0, -1, 0] = next_pred  # 新しい予測値を追加

    # 予測データをデータフレーム化
    forecast_index = pd.date_range(start=date_index[-1] + pd.Timedelta(days=1), periods=28, freq="D")
    forecast_df = pd.DataFrame({"Date": forecast_index, "Forecast": forecast})

    # 予測結果を保存
    forecast_df.to_csv("../data/lstm_forecast_all_clinics.csv", index=False)

    # 結果を可視化
    plt.figure(figsize=(15, 7))
    plt.plot(date_index[-len(y_test):], y_test, label="Actual", linewidth=2)
    plt.plot(date_index[-len(y_test):], y_pred_test, label="Predicted", linewidth=2)
    plt.plot(forecast_index, forecast, label="Forecast (Next 28 Days)", linestyle="dashed", linewidth=2)
    plt.legend(fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Counseled Count", fontsize=12)
    plt.title("LSTM Forecast with Improved Architecture", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    lstm_forecast()

if __name__ == "__main__":
    main()
