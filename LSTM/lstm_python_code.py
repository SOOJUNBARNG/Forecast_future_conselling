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
end_data = "2025-02-28"
start_data = pd.to_datetime(start_data)
end_data = pd.to_datetime(end_data)
data_start_date = start_data

def prepare_lstm_data(df, look_back=7):
    """LSTM用に時系列データを準備"""
    # まず基本データを取得
    df = data_process_group(df, data_start_date, end_data)
    
    # 日付をdatetimeに変換
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
    
    # 季節性の特徴を追加
    df["month"] = df.index.month
    df["month_mean"] = df.groupby("month")[target_col].transform("mean")
    
    # 週末フラグを追加
    df["is_weekend"] = df["day"].isin([5, 6]).astype(int)
    
    # 祝日フラグと週末フラグの組み合わせ
    df["holiday_or_weekend"] = (df["national_holiday"] | df["is_weekend"]).astype(int)
    
    # NaNを削除（最初の数行はラグの影響でNaNが入る）
    df.dropna(inplace=True)
    
    # 説明変数と目的変数を分割
    feature_cols.extend([
        "rolling_mean_7", "rolling_std_7", "day_mean", "week_of_month_mean",
        "month_mean", "is_weekend", "holiday_or_weekend"
    ])
    
    # 特徴量の準備
    X = df[feature_cols + [f"lag_{i}" for i in range(1, look_back + 1)]].values
    y = df[target_col].values
    
    # データを0-1にスケーリング
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
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
        
        for future_date in future_dates:
            # 日付関連の特徴を更新
            day = future_date.weekday()
            month = future_date.month
            week_of_month = (future_date.day + future_date.replace(day=1).weekday()) // 7 + 1
            
            # 祝日フラグの更新（カレンダーデータから取得する必要があります）
            # ここでは仮の値を使用
            national_holiday = 0  # カレンダーデータから取得する必要があります
            clinic_holiday = 0    # カレンダーデータから取得する必要があります
            
            # 週末フラグの更新
            is_weekend = int(day in [5, 6])
            holiday_or_weekend = int(national_holiday or is_weekend)
            
            # 新しい特徴量ベクトルを作成
            try:
                # 基本特徴量
                new_features = np.array([
                    national_holiday, clinic_holiday, day, week_of_month
                ])
                
                # 追加の特徴量（存在する場合のみ追加）
                if current_data.shape[1] > 4:
                    # 移動平均と標準偏差
                    rolling_mean = current_data[-1, 4] if current_data.shape[1] > 4 else 0
                    rolling_std = current_data[-1, 5] if current_data.shape[1] > 5 else 0
                    day_mean = current_data[-1, 6] if current_data.shape[1] > 6 else 0
                    week_mean = current_data[-1, 7] if current_data.shape[1] > 7 else 0
                    month_mean = current_data[-1, 8] if current_data.shape[1] > 8 else 0
                    
                    new_features = np.append(new_features, [
                        rolling_mean, rolling_std, day_mean, week_mean,
                        month_mean, is_weekend, holiday_or_weekend
                    ])
                else:
                    # 基本特徴量のみの場合、残りを0で埋める
                    new_features = np.append(new_features, [0] * 7)
                
            except IndexError as e:
                print(f"Warning: Could not access all features. Using default values. Error: {e}")
                # 基本特徴量のみを使用
                new_features = np.array([
                    national_holiday, clinic_holiday, day, week_of_month,
                    0, 0, 0, 0, 0, is_weekend, holiday_or_weekend
                ])
            
            # データを更新
            current_data = np.roll(current_data, shift=-1, axis=0)
            current_data[-1] = new_features
            
            # 予測用のデータを追加
            forecast_data.append(current_data)
        
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

def main():
    lstm_forecast()

if __name__ == "__main__":
    main()
