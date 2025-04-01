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
start_data = "2024-01-10"
start_data = pd.to_datetime(start_data)
current_date = pd.to_datetime(datetime.today().date())
current_date = pd.to_datetime("2025-03-01")
data_start_date = pd.to_datetime(f"{start_data}")

def arima_output():
    df = pd.read_csv("../data/counseling_count_group.csv")
    df = data_process_group(df, data_start_date, current_date)
    df.to_csv("hello.csv", index=False)
    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Define ARIMA parameters
    all_forecasts = []  # Store forecasts for all clinics

    # Set date as index
    df.set_index("date", inplace=True)

    # Convert holidays to binary flags
    df[["national_holiday", "clinic_holiday"]] = df[
        ["national_holiday", "clinic_holiday"]
    ].applymap(lambda x: 1 if x else 0)

    # Define target variable
    y = df.loc[df.index <= pd.to_datetime(current_date), "counseled"]

    # Exogenous variables (holiday flags)
    exog = df.loc[
        df.index <= pd.to_datetime(current_date),
        ["national_holiday", "clinic_holiday", "day", "week_of_month"],
    ]
    forecast_exog = df.loc[
        df.index > pd.to_datetime(current_date),
        ["national_holiday", "clinic_holiday", "day", "week_of_month"],
    ][:28]

    # Find the best ARIMA (p, d, q) with exogenous variables
    # {'p': 9, 'd': 1, 'q': 10} MAE: 146.07, RMSE: 222.10 今までベスト  
    #  5, 0, 5 / MAE: 209.59, RMSE: 271.63 / あいまい
    # 'p': 8, 'd': 0, 'q': 10 / MAE: 142.53, RMSE: 219.06
    best_p, best_d, best_q = 8, 0, 10
    model = ARIMA(y, order=(best_p, best_d, best_q), exog=exog)
    arima_result = model.fit()

    # Print model summary
    # print(arima_result.summary())

    # Forecast next 28 days
    forecast = arima_result.get_forecast(steps=28, exog=forecast_exog)
    # print(forecast.summary())
    # print(forecast.columns)

    # Get confidence intervals
    forecast_index = pd.date_range(
        start=y.index[-1] + pd.Timedelta(days=1), periods=28, freq="D"
    )
    forecast_mean = forecast.predicted_mean.astype(int)
    forecast_ci = forecast.conf_int()
    forecast_top = (
        forecast_ci.iloc[:, 1].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
    )
    forecast_bot = (
        forecast_ci.iloc[:, 0].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
    )

    # Store forecast results
    forecast_df = pd.DataFrame(
        {
            "Date": forecast_index,
            "Forecast": (forecast_top + forecast_bot) / 2,
            "Forecast 95% Top": forecast_top,
            "Forecast 95% Bot": forecast_bot,
        }
    )
    all_forecasts.append(forecast_df)

    # In-sample predictions
    y_pred = arima_result.predict(start=0, end=len(y) - 1)

    # Calculate errors
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Save all forecasts to a single CSV
    final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
    final_forecast_df.to_csv("../data/arima_forecast_all_clinics.csv", index=False)

def main():
    arima_output()


if __name__ == "__main__":
    main()
