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
current_date = datetime.today().date()

def before_sarima():
    # Load the dataset
    df = pd.read_csv("../data/counseling_count.csv")
    df = df[["クリニックID","クリニック名","日付","counseled"]]
    df = df.rename(columns={
        "クリニックID": "clinic_id",
        "クリニック名":"clinic_name",
        "日付":"date",
    })

    df = df[df["date"] > "2023-04-01"]

    df_clinic_unique = df[["clinic_id", "clinic_name"]].drop_duplicates()

    calendar_df = pd.read_csv("../data/calender.csv")
    calendar_df = calendar_df[["日付", "祝日フラグ", "TCB休診フラグ"]]
    calendar_df = calendar_df.rename(columns={
        "日付":"date",
        "祝日フラグ":"holiday_flag",
        "TCB休診フラグ":"tcb_holiday_flag",
    })
    calendar_df["date"] = pd.to_datetime(calendar_df["date"]).dt.strftime("%Y-%m-%d")
    cross_df = df_clinic_unique.merge(calendar_df, how="cross")

    cross_df_counsel = cross_df.merge(df, on=["clinic_id", "clinic_name", "date"], how="left")
    cross_df_counsel["counseled"] = cross_df_counsel["counseled"].fillna(0)

    # Display results
    print(cross_df_counsel.columns)
    print(cross_df_counsel.head(10))


    rest_day_df = pd.read_csv("../data/clinic_rest_day.csv")
    rest_day_df["name"] = rest_day_df["name"] + "院"
    rest_day_df = rest_day_df.rename(columns={
        "name":"clinic_name",
        "close_date":"date",
    })

    df_calender_rest = pd.merge(cross_df_counsel, rest_day_df, on=["clinic_name", "date"], how="left")
    df_calender_rest["status"] = df_calender_rest["status"].fillna(False)

    # Convert the date column to datetime format
    df_calender_rest["date"] = pd.to_datetime(df_calender_rest["date"])
    df_calender_rest = df_calender_rest[
        (df_calender_rest["date"] >= pd.Timestamp("2023-04-01")) &
        # (df_calender_rest["date"] <= pd.Timestamp(current_date) + pd.Timedelta(days=14))
        (df_calender_rest["date"] <= pd.Timestamp("2025-02-09") + pd.Timedelta(days=14))
    ]

    # 11,池袋駅前院,202502,2025-02-02,2025-02-08,17467130.9091,65,268725.09090909
    # 11,池袋駅前院,202502,2025-02-09,2025-02-09,6685830.9091,47,142251.72147002

    df_calender_rest = df_calender_rest.reset_index()

    df_calender_rest["national_holiday"] = df_calender_rest["holiday_flag"].apply(lambda x: 0 if x is False else 1)
    df_calender_rest["tcb_holiday"] = df_calender_rest.apply(
        lambda row: 0 if row["tcb_holiday_flag"] is False and row["tcb_holiday_flag"] is False else 1, 
        axis=1
    )
    df_before_sarima = df_calender_rest[["clinic_id","clinic_name", "date", "national_holiday", "tcb_holiday", "counseled"]]

    print(df_before_sarima.columns)
    print(df_before_sarima.head(10))

    return df_before_sarima


def sarima_output():
    df = before_sarima()  # Get preprocessed data

    df.to_csv("nyan.csv", index=False)

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Define SARIMA parameters
    best_params = {'p': 0, 'd': 1, 'q': 2, 'P': 3, 'D': 2, 'Q': 3}
    p, d, q = best_params["p"], best_params["d"], best_params["q"]
    P, D, Q, S = best_params["P"], best_params["D"], best_params["Q"], 7  # Weekly seasonality

    all_forecasts = []  # Store forecasts for all clinics

    # Loop through each clinic
    for clinic_id in df["clinic_id"].unique():
        df_clinic = df[df["clinic_id"] == clinic_id].copy()

        # Set date as index
        df_clinic.set_index("date", inplace=True)

        # Convert holidays to binary flags
        df_clinic[["national_holiday", "tcb_holiday"]] = df_clinic[["national_holiday", "tcb_holiday"]].applymap(lambda x: 1 if x else 0)

        # Define exogenous variables
        exog = df_clinic[["national_holiday", "tcb_holiday"]]
        y = df_clinic["counseled"]

        # Stationarity test
        result = adfuller(y.dropna())
        print(f"Clinic {clinic_id} - ADF Statistic: {result[0]}, p-value: {result[1]}")

        if result[1] > 0.05:
            print(f"Clinic {clinic_id}: Data is NOT stationary. Differencing is needed.")
        else:
            print(f"Clinic {clinic_id}: Data is stationary.")

        # Fit SARIMA model
        model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, S),
                        enforce_stationarity=False, enforce_invertibility=False, exog=exog)
        sarima_result = model.fit()

        # Print model summary
        print(sarima_result.summary())

        # Forecast next 14 days
        forecast_exog = exog.iloc[-14:].copy()
        forecast = sarima_result.get_forecast(steps=14, exog=forecast_exog)

        # Get confidence intervals
        forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=14, freq="D")
        forecast_mean = forecast.predicted_mean.astype(int)
        forecast_ci = forecast.conf_int()
        forecast_top = forecast_ci.iloc[:, 1].astype(int)
        forecast_bot = forecast_ci.iloc[:, 0].astype(int)

        # Store forecast results
        forecast_df = pd.DataFrame({
            "clinic_id": clinic_id,
            "Date": forecast_index,
            "Forecast": forecast_mean,
            "Forecast 95% Top": forecast_top,
            "Forecast 95% Bot": forecast_bot
        })
        all_forecasts.append(forecast_df)

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(y, label="Actual", color="blue")
        plt.plot(forecast_index, forecast_mean, label="Forecast", color="red")
        plt.fill_between(forecast_index, forecast_bot, forecast_top, color="pink", alpha=0.3)
        plt.xlabel("Date")
        plt.ylabel("Counseled")
        plt.title(f"SARIMA Forecast for Clinic {clinic_id}")
        plt.legend()
        plt.grid()
        plt.show()

        # In-sample predictions
        y_pred = sarima_result.predict(start=0, end=len(y)-1)

        # Calculate errors
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        print(f"Clinic {clinic_id} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Save all forecasts to a single CSV
    final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
    final_forecast_df.to_csv("../data/sarima_forecast_all_clinics.csv", index=False)

def main():
    sarima_output()

if __name__ == "__main__":
    main()