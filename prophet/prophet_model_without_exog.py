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
model.fit(df_prophet)

# --- Create future dataframe
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# --- Evaluation (only for overlapping dates)
merged = forecast.set_index("ds").join(df_prophet.set_index("ds"))
merged = merged.to_csv("merged.csv")
# 両方に非欠損値がある行だけフィルタリング
valid_rows = merged[["y", "yhat"]].dropna()
mae = mean_absolute_error(valid_rows["y"], valid_rows["yhat"])
print("Prophet MAE (no exog):", mae)
# --- Plot
model.plot(forecast)
