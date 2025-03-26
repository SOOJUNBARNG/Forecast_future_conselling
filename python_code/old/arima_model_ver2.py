# データ操作ライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 統計ライブラリ
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Load the dataset


df = pd.read_csv("../data/counseling_count.csv")


calendar_df = pd.read_csv("../data/calender.csv")
rest_day_df = pd.read_csv("../data/clinic_rest_day.csv")

# df = df.groupby("日付")["counseled"].sum().reset_index()
# df = df[["日付", "counseled"]]


# Convert the date column to datetime
df["日付"] = pd.to_datetime(df["日付"])

# Sort by date
df = df.sort_values("日付")
df["is_holiday"] = 0  # Default to 0


# Set date as index
df.set_index("日付", inplace=True)

exog = df[["clinic_holiday", "national_holiday"]]

# Keep only the target variable (counseled)
y = df["counseled"]

result = adfuller(y.dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] > 0.05:
    print("Data is NOT stationary. Differencing is needed.")
else:
    print("Data is stationary.")


best_params = {"p": 0, "d": 1, "q": 2, "P": 3, "D": 2, "Q": 3}

p, d, q = best_params["p"], best_params["d"], best_params["q"]
P, D, Q, S = (
    best_params["P"],
    best_params["D"],
    best_params["Q"],
    7,
)  # S=7 (週間の周期性)

# Define SARIMA model (p, d, q) x (P, D, Q, S)
model = SARIMAX(
    y,
    order=(p, d, q),
    seasonal_order=(P, D, Q, S),
    enforce_stationarity=False,
    enforce_invertibility=False,
    exog=exog,
)

# Fit the model
sarima_result = model.fit()

# Print summary
print(sarima_result.summary())

# Forecast next 14 days
forecast = sarima_result.get_forecast(steps=14)

# Get confidence intervals
forecast_index = pd.date_range(
    start=y.index[-1] + pd.Timedelta(days=1), periods=14, freq="D"
)
forecast_mean = forecast.predicted_mean.astype(int)  # 小数点を整数に変換
forecast_ci = forecast.conf_int()
forecast_top = forecast_ci.iloc[:, 1].astype(int)  # 上限
forecast_bot = forecast_ci.iloc[:, 0].astype(int)  # 下限

# Plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(y, label="Actual", color="blue")
plt.plot(forecast_index, forecast_mean, label="Forecast", color="red")
plt.fill_between(
    forecast_index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="pink",
    alpha=0.3,
)
plt.xlabel("Date")
plt.ylabel("Counseled")
plt.title("SARIMA Forecast for Counseled")
plt.legend()
plt.grid()
plt.show()

# In-sample predictions
y_pred = sarima_result.predict(start=0, end=len(y) - 1)

forecast_mean = forecast_mean.apply(lambda x: int(x))

# **予測データを CSV に保存**
forecast_df = pd.DataFrame(
    {
        "Date": forecast_index,
        "Forecast": forecast_mean,
        "Forecast 95% Top": forecast_top,
        "Forecast 95% Bot": forecast_bot,
    }
)
forecast_df.to_csv("../data/great.csv", index=False)

# Calculate errors
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
