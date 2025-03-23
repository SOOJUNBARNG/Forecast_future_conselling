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
current_date = datetime.today().date()


def plot_result(y, forecast, clinic_id):

    # Get confidence intervals
    forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=28, freq="D")
    forecast_mean = forecast.predicted_mean.astype(int)
    forecast_ci = forecast.conf_int()
    forecast_top = forecast_ci.iloc[:, 1].astype(int)
    forecast_bot = forecast_ci.iloc[:, 0].astype(int)

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


# def find_best_arima_params(y, exog=None, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
#     print(f"p_range: {p_range}, d_range: {d_range}, q_range: {q_range}")
#     """Find the best (p, d, q) for ARIMA using AIC."""
#     best_aic = float("inf")
#     best_params = None
    
#     for p, d, q in itertools.product(range(*p_range), range(*d_range), range(*q_range)):
#         try:
#             model = ARIMA(y, order=(p, d, q), exog=exog)  # Include exog
#             result = model.fit()
#             if result.aic < best_aic:
#                 best_aic = result.aic
#                 best_params = (p, d, q)
#         except:
#             continue  # Skip invalid models
    
#     return best_params


def before_arima():
    # Load the dataset
    df = pd.read_csv("../data/counseling_count.csv")
    df = df[["クリニックID","クリニック名","日付","counseled"]]
    df = df.rename(columns={
        "クリニックID": "clinic_id",
        "クリニック名":"clinic_name",
        "日付":"date",
    })

    df = df[df["date"] > "2025-01-10"]

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
    print(cross_df_counsel.index[cross_df_counsel.index.duplicated()])
    # print(cross_df_counsel.columns)
    # print(cross_df_counsel.head(10))


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
        (df_calender_rest["date"] >= pd.Timestamp("2024-03-01")) &
        # (df_calender_rest["date"] <= pd.Timestamp(current_date) + pd.Timedelta(days=14))
        (df_calender_rest["date"] <= pd.to_datetime(current_date) + pd.Timedelta(days=28))
    ]

    df_calender_rest = df_calender_rest.reset_index()
    # df_calender_rest.to_csv("check_for_great.csv", index=False)

    df_calender_rest["national_holiday"] = df_calender_rest["holiday_flag"].apply(lambda x: 0 if x is False else 1)
    df_calender_rest["tcb_holiday"] = df_calender_rest.apply(
        lambda row: 0 if row["tcb_holiday_flag"] is False and row["status"] is False else 1, 
        axis=1
    )
    df_before_arima = df_calender_rest[["clinic_id","clinic_name", "date", "national_holiday", "tcb_holiday", "counseled"]]
    df_before_arima = df_before_arima.rename(columns={
        "tcb_holiday":"clinic_holiday",
        })
    df_before_arima["day_of_week"] = df_before_arima["date"].dt.dayofweek
    # df_before_sarima["day_of_week"] = df_before_sarima["date"].dt.dayofweek.map(
    #     {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
    # )

    print(df_before_arima.index[df_before_arima.index.duplicated()])

    return df_before_arima


def arima_output():
    df = before_arima()  # Get preprocessed data

    # df.to_csv("nyan.csv", index=False)

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Define ARIMA parameters
    all_forecasts = []  # Store forecasts for all clinics

    # Loop through each clinic
    for clinic_id in df["clinic_id"].unique():
        df_clinic = df[df["clinic_id"] == clinic_id].copy()

        # Set date as index
        df_clinic.set_index("date", inplace=True)

        # Convert holidays to binary flags
        df_clinic[["national_holiday", "clinic_holiday"]] = df_clinic[["national_holiday", "clinic_holiday"]].applymap(lambda x: 1 if x else 0)

        # Define target variable
        y = df_clinic.loc[df_clinic.index <= pd.to_datetime(current_date), "counseled"]

        # Exogenous variables (holiday flags)
        exog = df_clinic.loc[df_clinic.index <= pd.to_datetime(current_date), ["national_holiday", "clinic_holiday","day_of_week"]]
        forecast_exog = df_clinic.loc[df_clinic.index > pd.to_datetime(current_date), ["national_holiday", "clinic_holiday","day_of_week"]][:28]

        # Find the best ARIMA (p, d, q) with exogenous variables
        # {'p': 5, 'd': 1, 'q': 2}.
        best_p, best_d, best_q = 5, 1, 2
        model = ARIMA(y, order=(best_p, best_d, best_q), exog=exog)
        arima_result = model.fit()

        # Print model summary
        # print(arima_result.summary())

        # Forecast next 14 days
        forecast = arima_result.get_forecast(steps=28, exog=forecast_exog)
        # print(forecast.summary())
        # print(forecast.columns)

        # Get confidence intervals
        forecast_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=28, freq="D")
        forecast_mean = forecast.predicted_mean.astype(int)
        forecast_ci = forecast.conf_int()
        forecast_top = forecast_ci.iloc[:, 1].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
        forecast_bot = forecast_ci.iloc[:, 0].fillna(0).replace([np.inf, -np.inf], 0).astype(int)

        # Store forecast results
        forecast_df = pd.DataFrame({
            "clinic_id": clinic_id,
            "Date": forecast_index,
            "Forecast": (forecast_top + forecast_bot) / 2,
            "Forecast 95% Top": forecast_top,
            "Forecast 95% Bot": forecast_bot
        })
        all_forecasts.append(forecast_df)

        # In-sample predictions
        y_pred = arima_result.predict(start=0, end=len(y)-1)

        # Calculate errors
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        print(f"Clinic {clinic_id} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Save all forecasts to a single CSV
    final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
    final_forecast_df.to_csv("../data/arima_forecast_all_clinics.csv", index=False)

    # Group forecast by Date and sum the forecast values
    group_forecast = final_forecast_df.groupby("Date").agg({
        "Forecast": "sum",
        "Forecast 95% Top": "sum",
        "Forecast 95% Bot": "sum"
    }).reset_index()

    group_forecast.to_csv("../data/arima_forecast.csv", index=False)

def main():
    arima_output()

if __name__ == "__main__":
    main()