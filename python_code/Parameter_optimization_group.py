# データ操作ライブラリ
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

# 統計ライブラリ
import optuna
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

start_data = "2024-01-10"
start_data = pd.to_datetime(start_data)

# Load dataset
file_path = "../data/counseling_count_group.csv"
df = pd.read_csv(file_path)
df["日付"] = pd.to_datetime(df["日付"])
df = df[df["日付"] >= pd.Timestamp(f"{start_data}")]
df = df.groupby("日付")["counseled"].sum().reset_index()
df.set_index("日付", inplace=True)

current_date = pd.to_datetime(datetime.today().date())


def data_process():
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
    # df_before_sarima["day_of_week"] = df_before_sarima["date"].dt.dayofweek.map(
    #     {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
    # )
    data_process["day_of_week"] = data_process.apply(
        lambda x: 5 if x["national_holiday"] == 1 else x["day_of_week"], axis=1
    )
    data_process = data_process[data_process["date"] >= pd.Timestamp(f"{start_data}")]

    print(data_process.index[data_process.index.duplicated()])

    return data_process


def arima_objective(trial):
    """Objective function for Optuna optimization."""
    # {'p': 9, 'd': 1, 'q': 10}
    p = trial.suggest_int("p", 0, 10)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 10)

    data_process_df = pd.DataFrame(data_process())
    print(data_process_df.columns)
    data_process_df["date"] = pd.to_datetime(data_process_df["date"], errors="coerce")
    data_process_df.reset_index(inplace=True)
    data_process_df.dropna(subset=["date"], inplace=True)
    data_process_df.set_index("date", inplace=True)

    # Convert date column to datetime
    current_date = pd.to_datetime(datetime.today().date())
    df_exog = data_process_df.copy()

    # Define target variable
    y = data_process_df.loc[data_process_df.index <= current_date, "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[
        df_exog.index <= current_date,
        ["national_holiday", "clinic_holiday", "day_of_week"],
    ]

    try:
        model = ARIMA(
            y,
            order=(p, d, q),
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit()
        return result.aic  # Minimize AIC
    except:
        return float("inf")  # Return high value if fitting fails


def optuna_arima():
    # Optuna で最適化
    study = optuna.create_study(direction="minimize")
    study.optimize(arima_objective, n_trials=300)  # 50回試行

    # 最適なハイパーパラメータを取得
    best_params = study.best_params
    # print("Best ARIMA Parameters:", best_params)

    data_process_df = pd.DataFrame(data_process())
    print(data_process_df.columns)
    data_process_df["date"] = pd.to_datetime(data_process_df["date"], errors="coerce")
    data_process_df.reset_index(inplace=True)
    data_process_df.dropna(subset=["date"], inplace=True)
    data_process_df.set_index("date", inplace=True)

    # Convert date column to datetime
    current_date = pd.to_datetime(datetime.today().date())
    df_exog = data_process_df.copy()

    # Define target variable
    y = data_process_df.loc[data_process_df.index <= current_date, "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[
        df_exog.index <= current_date,
        ["national_holiday", "clinic_holiday", "day_of_week"],
    ]

    best_model = ARIMA(
        y,
        order=(best_params["p"], best_params["d"], best_params["q"]),
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    best_result = best_model.fit()
    # print("Best ARIMA AIC:", best_result.aic)

    return best_params


# 目的関数の定義
def sarima_objective(trial):
    # Non-seasonal parameters
    # {'p': 5, 'd': 1, 'q': 2}.
    p = trial.suggest_int("p", 9, 9)  # Increased range
    d = trial.suggest_int("d", 1, 1)
    q = trial.suggest_int("q", 10, 10)  # Increased range

    # Seasonal parameters
    P = trial.suggest_int("P", 0, 10)  # Increased range
    D = trial.suggest_int("D", 0, 2)
    Q = trial.suggest_int("Q", 0, 10)  # Increased range
    # S = trial.suggest_int("S", 0, 100)  # Weekly, Monthly, Yearly
    S = trial.suggest_categorical("S", [365])  # Weekly or monthly seasonality

    data_process_df = pd.DataFrame(data_process())
    print(data_process_df.columns)
    data_process_df["date"] = pd.to_datetime(data_process_df["date"], errors="coerce")
    data_process_df.reset_index(inplace=True)
    data_process_df.dropna(subset=["date"], inplace=True)
    data_process_df.set_index("date", inplace=True)

    # Convert date column to datetime
    current_date = pd.to_datetime(datetime.today().date())
    df_exog = data_process_df.copy()

    # Define target variable
    y = data_process_df.loc[data_process_df.index <= current_date, "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[
        df_exog.index <= current_date,
        ["national_holiday", "clinic_holiday", "day_of_week"],
    ]

    try:
        # SARIMAX モデルの構築と学習
        model = SARIMAX(
            y,
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


def optuna_sarima():
    # Optuna で最適化
    study = optuna.create_study(direction="minimize")
    study.optimize(sarima_objective, n_trials=300)  # 50回試行

    # 最適なハイパーパラメータを取得
    best_params = study.best_params
    # print("Best SARIMA Parameters:", best_params)

    data_process_df = pd.DataFrame(data_process())
    print(data_process_df.columns)
    data_process_df["date"] = pd.to_datetime(data_process_df["date"], errors="coerce")
    data_process_df.reset_index(inplace=True)
    data_process_df.dropna(subset=["date"], inplace=True)
    data_process_df.set_index("date", inplace=True)

    # Convert date column to datetime
    current_date = pd.to_datetime(datetime.today().date())
    df_exog = data_process_df.copy()

    # Define target variable
    y = data_process_df.loc[data_process_df.index <= current_date, "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[
        df_exog.index <= current_date,
        ["national_holiday", "clinic_holiday", "day_of_week"],
    ]

    # 最適パラメータでモデルを再学習
    best_model = SARIMAX(
        y,
        order=(best_params["p"], best_params["d"], best_params["q"]),
        seasonal_order=(best_params["P"], best_params["D"], best_params["Q"], 7),
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    best_result = best_model.fit(disp=False)
    # print("Best SARIMA AIC:", best_result.aic)

    return best_params


if __name__ == "__main__":
    # print(optuna_arima())
    print(optuna_sarima())
