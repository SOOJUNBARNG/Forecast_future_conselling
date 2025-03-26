# データ操作ライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 統計ライブラリ
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# 時間関連ライブラリ
from datetime import datetime, timedelta

# Get the current date
start_data = "2024-01-10"
start_data = pd.to_datetime(start_data)
current_date = pd.to_datetime(datetime.today().date())
current_date = pd.to_datetime("2025-03-01")


def plot_result(y, forecast, clinic_id):

    # Get confidence intervals
    forecast_index = pd.date_range(
        start=y.index[-1] + pd.Timedelta(days=1), periods=14, freq="D"
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


def before_sarima():
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
    data_process["day_of_week"] = data_process["date"].dt.dayofweek
    data_process["day_of_week"] = data_process.apply(
        lambda x: 5 if x["national_holiday"] == 1 else x["day_of_week"], axis=1
    )
    # df_before_sarima["day_of_week"] = df_before_sarima["date"].dt.dayofweek.map(
    #     {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
    # )
    data_process = data_process[data_process["date"] >= pd.Timestamp(f"{start_data}")]

    print(data_process.index[data_process.index.duplicated()])

    return data_process


def sarima_output():
    df = before_sarima()  # Get preprocessed data

    # df.to_csv("check_for_normal.csv", index=False)

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])
    df.index = pd.DatetimeIndex(df.index).to_period("D")  # Daily frequency

    # 18 /  {'p': 5, 'd': 1, 'q': 2, 'P': 0, 'D': 0, 'Q': 1, 'S': 69}
    # {'p': 5, 'd': 1, 'q': 2, 'P': 0, 'D': 2, 'Q': 0, 'S': 81}

    # Define SARIMA parameters
    best_params = {"p": 9, "d": 1, "q": 10, "P": 0, "D": 1, "Q": 2, "S": 365}
    p, d, q = best_params["p"], best_params["d"], best_params["q"]
    P, D, Q, S = (
        best_params["P"],
        best_params["D"],
        best_params["Q"],
        best_params["S"],
    )  # Yearly seasonality

    all_forecasts = []  # Store forecasts for all clinics


    # Set date as index
    df.set_index("date", inplace=True)

    # Convert holidays to binary flags
    df[["national_holiday", "clinic_holiday"]] = df[["national_holiday", "clinic_holiday"]].applymap(lambda x: 1 if x else 0)

    # Define target variable
    y = df.loc[df.index <= pd.to_datetime(current_date), "counseled"]

    # Exogenous variables (holiday flags)
    exog = df.loc[df.index <= pd.to_datetime(current_date),["national_holiday", "clinic_holiday", "day_of_week"],]
    forecast_exog = df.loc[df.index > pd.to_datetime(current_date),["national_holiday", "clinic_holiday", "day_of_week"],][:28]


    # Check if indices match
    if not y.index.equals(exog.index):
        print("Warning: Indices do not match between y and exog.")

    # Proceed with SARIMAX model fitting
    model = SARIMAX(
        y,
        order=(p, d, q),
        seasonal_order=(P, D, Q, S),
        enforce_stationarity=False,
        enforce_invertibility=False,
        exog=exog,
    )
    sarima_result = model.fit()

    # Print model summary
    print(sarima_result.summary())

    # Forecast next 14 days
    # Prepare forecast_exog for the next 14 days
    if len(exog) >= 28:
        forecast_exog = exog.iloc[-28:].copy()
    else:
        # If there are less than 14 rows, repeat the last available rows
        forecast_exog = pd.concat(
            [exog.tail(1)] * (28 - len(exog)), ignore_index=True
        )

    # Forecast the next 14 days
    forecast = sarima_result.get_forecast(steps=28, exog=forecast_exog)

    # Get confidence intervals
    forecast_index = pd.date_range(
        start=y.index[-1] + pd.Timedelta(days=1), periods=28, freq="D"
    )
    forecast_mean = forecast.predicted_mean.astype(int)
    forecast_ci = forecast.conf_int()
    forecast_top = forecast_ci.iloc[:, 1].astype(int)
    forecast_bot = forecast_ci.iloc[:, 0].astype(int)

    # Store forecast results
    forecast_df = pd.DataFrame(
        {
            "Date": forecast_index,
            "Forecast": forecast_mean,
            "Forecast 95% Top": forecast_top,
            "Forecast 95% Bot": forecast_bot,
        }
    )
    all_forecasts.append(forecast_df)

    # In-sample predictions
    y_pred = sarima_result.predict(start=0, end=len(y) - 1)

    # Calculate errors
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # print
    # plot_result(y, forecast, clinic_id)

    # Save all forecasts to a single CSV
    final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
    final_forecast_df.to_csv("../data/sarima_forecast_all_clinics.csv", index=False)

    # Group forecast by Date and sum the forecast values
    group_forecast = (
        final_forecast_df.groupby("Date")
        .agg({"Forecast": "sum", "Forecast 95% Top": "sum", "Forecast 95% Bot": "sum"})
        .reset_index()
    )

    group_forecast.to_csv("../data/sarima_forecast_group.csv", index=False)


def main():
    sarima_output()


if __name__ == "__main__":
    main()
