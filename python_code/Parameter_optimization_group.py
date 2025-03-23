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
file_path = "../data/counseling_count_group.csv"
df = pd.read_csv(file_path)
df["日付"] = pd.to_datetime(df["日付"])
df = df[
    df["日付"] >= pd.Timestamp("2025-01-10") 
]
df = df.groupby("日付")["counseled"].sum().reset_index()
df.set_index("日付", inplace=True)

current_date = datetime.today().date()


def data_process():
    # Load the dataset
    df = pd.read_csv("../data/counseling_count_group.csv")
    df = df[["日付","counseled"]]
    df = df.rename(columns={
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
    data_process = df_calender_rest[["clinic_id","clinic_name", "date", "national_holiday", "tcb_holiday", "counseled"]]
    data_process = data_process.rename(columns={
        "tcb_holiday":"clinic_holiday",
        })
    data_process["day_of_week"] = data_process["date"].dt.dayofweek
    # df_before_sarima["day_of_week"] = df_before_sarima["date"].dt.dayofweek.map(
    #     {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
    # )

    print(data_process.index[data_process.index.duplicated()])

    return data_process


def arima_objective(trial):
    """Objective function for Optuna optimization."""
    p = trial.suggest_int("p", 0, 5)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 5)

    data_process_df = data_process()
    data_process_df["date"] = pd.to_datetime(data_process_df["date"])

    # Convert date column to datetime
    df_exog = data_process_df.copy()
    
    # Set date as index
    df_exog.set_index("date", inplace=True)

    # Convert holidays to binary flags
    df_exog[["national_holiday", "clinic_holiday"]] = df_exog[["national_holiday", "clinic_holiday"]].applymap(lambda x: 1 if x else 0)

    # Define target variable
    y = data_process_df.loc[data_process_df.index <= pd.to_datetime(current_date), "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[df_exog.index <= pd.to_datetime(current_date), ["national_holiday", "clinic_holiday","day_of_week"]]

    # # Define exogenous variables
    # exog_vars = ["feature1", "feature2"]  # Replace with actual column names
    # exog = df[exog_vars] if exog_vars else None

    try:
        model = ARIMA(df["counseled"], order=(p, d, q), exog=exog,
                      enforce_stationarity=False, enforce_invertibility=False)
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

    data_process_df = data_process()

    # Convert date column to datetime
    df_exog = data_process_df.copy()
    df_exog["date"] = pd.to_datetime(df_exog["date"])

    # Set date as index
    df_exog.set_index("date", inplace=True)

    # Convert holidays to binary flags
    df_exog[["national_holiday", "clinic_holiday"]] = df_exog[["national_holiday", "clinic_holiday"]].applymap(lambda x: 1 if x else 0)

    # Define target variable
    y = df_exog.loc[df_exog.index <= pd.to_datetime(current_date), "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[df_exog.index <= pd.to_datetime(current_date), ["national_holiday", "clinic_holiday","day_of_week"]]

    best_model = ARIMA(df["counseled"], order=(best_params["p"], best_params["d"], best_params["q"]),
                       exog=exog, enforce_stationarity=False, enforce_invertibility=False)
    
    best_result = best_model.fit()
    # print("Best ARIMA AIC:", best_result.aic)

    return best_params

# 目的関数の定義
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
    df_exog[["national_holiday", "clinic_holiday"]] = df_exog[["national_holiday", "clinic_holiday"]].applymap(lambda x: 1 if x else 0)

    # Define target variable
    y = df_exog.loc[df_exog.index <= pd.to_datetime(current_date), "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[df_exog.index <= pd.to_datetime(current_date), ["national_holiday", "clinic_holiday","day_of_week"]]

    try:
        # SARIMAX モデルの構築と学習
        model = SARIMAX(df["counseled"], 
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, S),
                        exog=exog,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
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

    data_process_df = data_process()

    # Convert date column to datetime
    df_exog = data_process_df.copy()
    df_exog["date"] = pd.to_datetime(df_exog["date"])

    # Set date as index
    df_exog.set_index("date", inplace=True)

    # Convert holidays to binary flags
    df_exog[["national_holiday", "clinic_holiday"]] = df_exog[["national_holiday", "clinic_holiday"]].applymap(lambda x: 1 if x else 0)

    # Define target variable
    y = df_exog.loc[df_exog.index <= pd.to_datetime(current_date), "counseled"]

    # Exogenous variables (holiday flags)
    exog = df_exog.loc[df_exog.index <= pd.to_datetime(current_date), ["national_holiday", "clinic_holiday","day_of_week"]]

    # 最適パラメータでモデルを再学習
    best_model = SARIMAX(df["counseled"],
                        order=(best_params["p"], best_params["d"], best_params["q"]),
                        seasonal_order=(best_params["P"], best_params["D"], best_params["Q"], 7),
                        exog=exog,
                        enforce_stationarity=False,
                        enforce_invertibility=False)

    best_result = best_model.fit(disp=False)
    # print("Best SARIMA AIC:", best_result.aic)

    return best_params



if __name__ == "__main__":
    print(optuna_arima())
    print(optuna_sarima())
