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

# Get the current date
start_data = "2024-01-10"
start_data = pd.to_datetime(start_data)
current_date = pd.to_datetime(datetime.today().date())
current_date = pd.to_datetime("2025-03-01")

def get_nth_week_of_month(date):
    first_day_of_month = date.replace(day=1)
    first_weekday = first_day_of_month.weekday()  # Monday = 0, Sunday = 6
    current_weekday = date.weekday()
    
    # Calculate the nth week of the month
    nth_week = (date.day + first_weekday) // 7 + 1
    return nth_week


def plot_result(y, forecast, clinic_id):

    # Get confidence intervals
    forecast_index = pd.date_range(
        start=y.index[-1] + pd.Timedelta(days=1), periods=28, freq="D"
    )
    forecast_mean = forecast.predicted_mean.astype(int)
    forecast_ci = forecast.conf_int()
    forecast_top = forecast_ci.iloc[:, 1].astype(int)
    forecast_bot = forecast_ci.iloc[:, 0].astype(int)

    plt.figure(figsize=(12, 6))
    plt.plot(y, label="Actual", color="blue")
    plt.plot(forecast_index, forecast_mean, label="Forecast", color="red")
    plt.fill_between(
        forecast_index, forecast_bot, forecast_top, color="pink", alpha=0.3
    )
    plt.xlabel("Date")
    plt.ylabel("Counseled")
    plt.title(f"SARIMA Forecast for Clinic {clinic_id}")
    plt.legend()
    plt.grid()
    plt.show()


def before_arima():
    # Load the dataset
    df = pd.read_csv("../data/counseling_count_group.csv")
    df = df[["日付", "counseled"]]
    df = df.rename(
        columns={
            "日付": "date",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    df = df[df["date"] > f"{start_data}"]

    calendar_df = pd.read_csv("../data/calender.csv")
    calendar_df = calendar_df[["日付", "祝日フラグ", "TCB休診フラグ"]]
    calendar_df = calendar_df.rename(
        columns={
            "日付": "date",
            "祝日フラグ": "holiday_flag",
            "TCB休診フラグ": "tcb_holiday_flag",
        }
    )
    calendar_df["date"] = pd.to_datetime(calendar_df["date"]).dt.strftime("%Y-%m-%d")
    cross_df = df.merge(calendar_df, on="date", how="outer")
    cross_df["counseled"] = cross_df["counseled"].fillna(0)

    # cross_df.to_csv("print_check_ver2.csv", index=False)

    # Display results
    print(cross_df.index[cross_df.index.duplicated()])
    cross_df = cross_df.reset_index()

    cross_df["national_holiday"] = cross_df["holiday_flag"].apply(
        lambda x: 0 if x is False else 1
    )
    cross_df["tcb_holiday"] = cross_df.apply(
        lambda row: 0 if row["tcb_holiday_flag"] is False else 1, axis=1
    )
    data_process = cross_df[["date", "national_holiday", "tcb_holiday", "counseled"]]
    data_process["date"] = pd.to_datetime(data_process["date"], errors="coerce")
    data_process = data_process.rename(
        columns={
            "tcb_holiday": "clinic_holiday",
        }
    )
    data_process["day"] = data_process["date"].dt.dayofweek
    data_process["week_of_month"] = data_process["date"].apply(lambda x: get_nth_week_of_month(x))
    data_process["day"] = data_process.apply(
        lambda x: 5 if x["national_holiday"] == 1 else x["day"], axis=1
    )
    # df_before_sarima["day_of_week"] = df_before_sarima["date"].dt.dayofweek.map(
    #     {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
    # )
    data_process = data_process[data_process["date"] >= pd.Timestamp(f"{start_data}")]

    print(data_process.index[data_process.index.duplicated()])

    return data_process


def arima_output():
    df = before_arima()  # Get preprocessed data
    df.to_csv("hello.csv")
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
    best_p, best_d, best_q = 9, 1, 10
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

    # Group forecast by Date and sum the forecast values
    group_forecast = (
        final_forecast_df.groupby("Date")
        .agg({"Forecast": "sum", "Forecast 95% Top": "sum", "Forecast 95% Bot": "sum"})
        .reset_index()
    )

    group_forecast.to_csv("../data/arima_forecast_group.csv", index=False)


def main():
    arima_output()


if __name__ == "__main__":
    main()
