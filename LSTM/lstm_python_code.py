# ファイル操作ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ統計ライブラリ
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
start_data = "2023-04-10"
end_data = "2025-02-28"
start_data = pd.to_datetime(start_data)
end_data = pd.to_datetime(end_data)
data_start_date = start_data

# 	date	national_holiday	counseled
# date	1.0	0.0020438204036200024	0.14642720944366858
# national_holiday	0.0020438204036200024	1.0	0.7151975888697979
# counseled	0.14642720944366858	0.7151975888697979	1.0
# day	-0.001197187150115733	0.6928931664157294	0.5037617844721227
# day_num	0.019668216172348686	-0.040861433796522365	0.1574701559873876
# 26_31_bin	0.0010639329480751532	-0.04733732093401027	0.13798798637616383
# month_group_one	0.11445529158716136	-0.011259190302512637	0.17249787140979553
# month_group_two	0.3019136089439945	0.022378493557617722	0.13631489524594892
# week_of_month	0.051708871150628735	-0.02811209604058727	0.13340869600954325
# counseled_lag1	0.1547958931766842	0.3340778154634682	0.3123114853574727

def prepare_lstm_data(df, look_back=7):
    """LSTM用に時系列データを準備"""
    # まず基本データを取得
    df = data_process_group(df, data_start_date, end_data)
    
    # 日付をdatetimeに変換
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    
    # 必要な列を取得
    target_col = "counseled"
    feature_cols = ["day", "national_holiday","week_of_month", "day_num", "26_31_bin", "month_group_one", "month_group_two"]   
    
    # NaNを削除（最初の数行はラグの影響でNaNが入る）
    df.dropna(inplace=True)
    
    # データを0-1にスケーリング
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    y = df[target_col].values
    
    print(f"Input shape: {X.shape}, Features: {len(feature_cols)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # LSTMの入力形式に変換（[サンプル数, タイムステップ, 特徴量数]）
    global look_back_count
    look_back_count = look_back + len(feature_cols)
    X = X.reshape((X.shape[0], look_back_count, -1))
    
    return X, y, scaler, df.index

def prepare_forecast_data(last_data, future_dates, scaler, look_back):
    """未来予測用のデータを準備"""
    try:
        forecast_data = []
        current_data = last_data.copy()

        return np.array(forecast_data)
    
    except Exception as e:
        print(f"Error in prepare_forecast_data: {e}")
        print(f"Current data shape: {current_data.shape}")
        print(f"Future dates: {future_dates}")
        raise

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
def lstm_forecast(data_start_date, end_data):
    df = pd.read_csv("../data/counseling_count_group.csv")
    # データの準備
    look_back = 7  # 過去7日間を使用
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
    
    # 未来の2週間を予測
    future_dates = pd.date_range(start=date_index[-1] + pd.Timedelta(days=1), periods=14, freq="D")
    forecast_data = prepare_forecast_data(X_test[-1], future_dates, scaler, look_back)
    forecast = model.predict(forecast_data)
    
    # 予測データをデータフレーム化
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast.flatten()
    })
    
    # 予測結果を保存
    forecast_df.to_csv("../data/lstm_forecast_all_clinics.csv", index=False)
    
    # 結果を可視化
    plt.figure(figsize=(15, 7))
    plt.plot(date_index[-len(y_test):], y_test, label="Actual", linewidth=2)
    plt.plot(date_index[-len(y_test):], y_pred_test, label="Predicted", linewidth=2)
    plt.plot(future_dates, forecast, label="Forecast (Next 14 Days)", linestyle="dashed", linewidth=2)
    plt.legend(fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Counseled Count", fontsize=12)
    plt.title("LSTM Forecast with Improved Architecture", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Get the current date
start_data = "2023-04-01"
start_data = pd.to_datetime(start_data)
current_date = current_date
data_start_date = pd.to_datetime(f"{start_data}")

# Get the current date
start_data = "2023-04-10"
end_data = "2025-02-28"
start_data = pd.to_datetime(start_data)
end_data = pd.to_datetime(end_data)
data_start_date = start_data

def main():
    lstm_forecast(data_start_date, end_data)

if __name__ == "__main__":
    main()
