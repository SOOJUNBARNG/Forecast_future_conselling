# データ操作ライブラリ
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

# 統計ライブラリ
import optuna
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load dataset
file_path = "../data/counseling_count.csv"
df = pd.read_csv(file_path)
df["日付"] = pd.to_datetime(df["日付"])
df = df[df["日付"] >= pd.Timestamp("2025-01-10")]
df = df.groupby("日付")["counseled"].sum().reset_index()
df.set_index("日付", inplace=True)

current_date = pd.to_datetime(datetime.today().date())


def data_process():
    # Load the dataset
    df = pd.read_csv("../data/counseling_count.csv")
    df = df[["クリニックID", "クリニック名", "日付", "counseled"]]
    df = df.rename(
        columns={
            "クリニックID": "clinic_id",
            "クリニック名": "clinic_name",
            "日付": "date",
        }
    )

    df = df[df["date"] > "2025-01-10"]

    df_clinic_unique = df[["clinic_id", "clinic_name"]].drop_duplicates()

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
    cross_df = df_clinic_unique.merge(calendar_df, how="cross")

    cross_df_counsel = cross_df.merge(
        df, on=["clinic_id", "clinic_name", "date"], how="left"
    )
    cross_df_counsel["counseled"] = cross_df_counsel["counseled"].fillna(0)

    # Display results
    print(cross_df_counsel.index[cross_df_counsel.index.duplicated()])
    # print(cross_df_counsel.columns)
    # print(cross_df_counsel.head(10))

    rest_day_df = pd.read_csv("../data/clinic_rest_day.csv")
    rest_day_df["name"] = rest_day_df["name"] + "院"
    rest_day_df = rest_day_df.rename(
        columns={
            "name": "clinic_name",
            "close_date": "date",
        }
    )

    df_calender_rest = pd.merge(
        cross_df_counsel, rest_day_df, on=["clinic_name", "date"], how="left"
    )
    df_calender_rest["status"] = df_calender_rest["status"].fillna(False)

    # Convert the date column to datetime format
    df_calender_rest["date"] = pd.to_datetime(df_calender_rest["date"])
    df_calender_rest = df_calender_rest[
        (df_calender_rest["date"] >= pd.Timestamp("2024-03-01"))
        &
        # (df_calender_rest["date"] <= pd.Timestamp(current_date) + pd.Timedelta(days=14))
        (
            df_calender_rest["date"]
            <= pd.to_datetime(current_date) + pd.Timedelta(days=28)
        )
    ]

    df_calender_rest = df_calender_rest.reset_index()
    # df_calender_rest.to_csv("check_for_great.csv", index=False)

    df_calender_rest["national_holiday"] = df_calender_rest["holiday_flag"].apply(
        lambda x: 0 if x is False else 1
    )
    df_calender_rest["tcb_holiday"] = df_calender_rest.apply(
        lambda row: (
            0 if row["tcb_holiday_flag"] is False and row["status"] is False else 1
        ),
        axis=1,
    )
    data_process = df_calender_rest[
        [
            "clinic_id",
            "clinic_name",
            "date",
            "national_holiday",
            "tcb_holiday",
            "counseled",
        ]
    ]
    data_process = data_process.rename(
        columns={
            "tcb_holiday": "clinic_holiday",
        }
    )
    data_process["day_of_week"] = data_process["date"].dt.dayofweek
    # df_before_sarima["day_of_week"] = df_before_sarima["date"].dt.dayofweek.map(
    #     {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
    # )

    print(data_process.index[data_process.index.duplicated()])

    data_process.to_csv("../data/nyan.csv", index=False)

    return data_process


def arima_objective(trial):
    """Objective function for Optuna optimization."""
    p = trial.suggest_int("p", 0, 5)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 5)

    # Data processing and date conversion
    data_process_df = data_process()
    data_process_df["date"] = pd.to_datetime(data_process_df["date"])

    # Ensure current_date is defined and in the correct format
    current_date = pd.to_datetime(
        datetime.today().date()
    )  # Convert current_date to Timestamp

    # Convert 'date' column to datetime and set it as the index for exog
    df_exog = data_process_df.copy()
    df_exog.set_index("date", inplace=True)

    # Convert holidays to binary flags using apply
    df_exog[["national_holiday", "clinic_holiday"]] = df_exog[
        ["national_holiday", "clinic_holiday"]
    ].apply(lambda x: x.astype(int))

    # Filter data for dates less than or equal to current_date
    filtered_df = data_process_df.loc[data_process_df["date"] <= current_date]

    # If no data available, return a high value for the trial
    if filtered_df.empty:
        print("No data available for the specified date range.")
        return float("inf")  # Return a high value if no data is available

    # Define target variable (counseled column)
    y = filtered_df["counseled"]

    # Filter exogenous variables (holidays and day of the week) based on the same date range
    exog = df_exog.loc[
        df_exog.index <= current_date,
        ["national_holiday", "clinic_holiday", "day_of_week"],
    ]

    # Align the indices of y and exog before fitting ARIMA model
    exog = exog.loc[exog.index.isin(y.index)]  # Make sure exog has the same index as y
    y = y.loc[exog.index]  # Filter y to have the same index as exog

    try:
        # Fit ARIMA model
        model = ARIMA(
            y,
            order=(p, d, q),
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit()
        return result.aic  # Minimize AIC
    except Exception as e:
        # If ARIMA fitting fails, return a high value
        print(f"Error in ARIMA fitting: {e}")
        return float("inf")


def sarima_objective(trial):
    # Non-seasonal parameters
    # {'p': 5, 'd': 1, 'q': 2}.
    p = trial.suggest_int("p", 0, 5)  # Increased range
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 5)  # Increased range

    # Seasonal parameters
    P = trial.suggest_int("P", 0, 5)  # Increased range
    D = trial.suggest_int("D", 0, 2)
    Q = trial.suggest_int("Q", 0, 5)  # Increased range
    # S = trial.suggest_int("S", 0, 100)  # Weekly, Monthly, Yearly
    S = trial.suggest_categorical("S", [7])  # Weekly or monthly seasonality

    data_process_df = data_process()

    # Convert date column to datetime
    df_exog = data_process_df.copy()
    df_exog["date"] = pd.to_datetime(df_exog["date"])

    # Set date as index
    df_exog.set_index("date", inplace=True)

    # Convert holidays to binary flags
    df_exog[["national_holiday", "clinic_holiday"]] = df_exog[
        ["national_holiday", "clinic_holiday"]
    ].applymap(lambda x: 1 if x else 0)

    # Define target variable
    y = df_exog.loc[df_exog.index <= pd.to_datetime(current_date), "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[
        df_exog.index <= pd.to_datetime(current_date),
        ["national_holiday", "clinic_holiday", "day_of_week"],
    ]

    try:
        # SARIMAX モデルの構築と学習
        model = SARIMAX(
            df["counseled"],
            order=(p, d, q),
            seasonal_order=(P, D, Q, S),
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)

        return result.aic  # AIC を最小化
    except:
        return np.inf  # エラー発生時は最悪値


def optimize_model(objective_function, n_trials=300):
    # General optimization function using Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_function, n_trials=n_trials)
    return study.best_params


if __name__ == "__main__":
    # Call optimize_model for both ARIMA and SARIMA
    best_arima_params = optimize_model(arima_objective)
    best_sarima_params = optimize_model(sarima_objective)

    print("Best ARIMA Params:", best_arima_params)
    print("Best SARIMA Params:", best_sarima_params)
