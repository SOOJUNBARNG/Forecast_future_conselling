# Time
from datetime import datetime

# Data manipulation
import pandas as pd

# ML
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

import sys
from pathlib import Path

# プロジェクトルートを `sys.path` に追加
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.print_output_in_matplotlib import plot_result
from utils.data_pre_process import data_process_group

# --- Set dates
start_data = "2023-04-01"
start_data = pd.to_datetime(start_data)
current_date = pd.to_datetime("2025-03-28")
data_start_date = pd.to_datetime(f"{start_data}")

# --- Data loading function
def data_output():
    df = pd.read_csv("../data/counseling_count_group.csv")
    df = data_process_group(df, data_start_date, current_date)
    df.to_csv("hello.csv", index=False)
    df["date"] = pd.to_datetime(df["date"])
    return df

# --- Prepare data
df_prophet = data_output()
df_prophet = df_prophet.rename(columns={
    "date": "ds",
    "counseled": "y"
})

# --- Fit Prophet model
model = Prophet()
model.add_regressor("day")
model.add_regressor("counseled_lag1")
model.add_regressor("counseled_lag7")
model.add_regressor("counseled_lag28")
model.fit(df_prophet)

# --- Create future dataframe
future = df_prophet[["ds"]].copy()
future = model.make_future_dataframe(periods=30)
# Add exogenous variables
future["day"] = future["ds"].dt.dayofweek

# Use last known lag values for future rows
last_known = df_prophet.set_index("ds").iloc[-1]
for col in ["counseled_lag1", "counseled_lag7", "counseled_lag28"]:
    future[col] = last_known[col]

# Fill past values from original df
future = future.merge(df_prophet[["ds", "day", "counseled_lag1", "counseled_lag7", "counseled_lag28"]],
                      on="ds", how="left", suffixes=("", "_past"))

# For existing past rows, use actual values
for col in ["day", "counseled_lag1", "counseled_lag7", "counseled_lag28"]:
    future[col] = future[f"{col}_past"].combine_first(future[col])
    future.drop(columns=[f"{col}_past"], inplace=True)

# Predict
forecast = model.predict(future)

# Evaluate
merged = forecast.set_index("ds").join(
    df_prophet.set_index("ds"),
    lsuffix="_forecast",
    rsuffix="_actual"
)
valid = merged[["y", "yhat"]].dropna()
mae = mean_absolute_error(valid["y"], valid["yhat"])
print("Prophet + exog MAE:", mae)

merged.to_csv("forecast_with_exog.csv")

