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
df = df[["クリニックID","クリニック名","日付","counseled"]]
df = df.rename(columns={
    "クリニックID": "clinic_id",
    "クリニック名":"clinic_name",
    "日付":"date",
})

calendar_df = pd.read_csv("../data/calender.csv")
calendar_df = calendar_df[["日付", "祝日フラグ", "TCB休診フラグ"]]
calendar_df = calendar_df.rename(columns={
    "日付":"date",
    "祝日フラグ":"holiday_flag",
    "TCB休診フラグ":"tcb_holiday_flag",
})
calendar_df["date"] = pd.to_datetime(calendar_df["date"]).dt.strftime("%Y-%m-%d")
df_calender = pd.merge( calendar_df, df,on="date", how="left")
df_calender = df_calender.sort_values(["clinic_id", "date"], ascending=True)
df_calender = df_calender.dropna(subset = ["clinic_id"]).reset_index()
# df_calender["counseled"] = df_calender["counseled"].astype(int)

df_calender = df_calender[["clinic_name", "date", "counseled","holiday_flag", "tcb_holiday_flag"]]

# Iterate over rows using iterrows() to properly modify values
for idx, row in df_calender.iterrows():
    # Check if tcb_holiday_flag is True or if counseled is null, 0, or empty
    if row["tcb_holiday_flag"] is True or pd.isnull(row["counseled"]) or row["counseled"] == 0 or row["counseled"] == "":
        df_calender.at[idx, "tcb_holiday_flag"] = True


print(df_calender.columns)
print(df_calender.head(10))

# Convert the date column to datetime
df_calender["date"] = pd.to_datetime(df_calender["date"])

# Sort by date
df_calender["holiday_flag"] = 0  # Default to 0

# Set date as index
df_calender.set_index("date", inplace=True)
df_calender[["holiday_flag", "tcb_holiday_flag"]] = df_calender[["holiday_flag", "tcb_holiday_flag"]].applymap(lambda x: 1 if x else 0)

exog = df_calender[["holiday_flag", "tcb_holiday_flag"]]

# Keep only the target variable (counseled)
y = df_calender["counseled"]

result = adfuller(y.dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] > 0.05:
    print("Data is NOT stationary. Differencing is needed.")
else:
    print("Data is stationary.")


best_params = {'p': 0, 'd': 1, 'q': 2, 'P': 3, 'D': 2, 'Q': 3}

p, d, q = best_params["p"], best_params["d"], best_params["q"]
P, D, Q, S = best_params["P"], best_params["D"], best_params["Q"], 7  # S=7 (週間の周期性)

# Define SARIMA model (p, d, q) x (P, D, Q, S)
model = SARIMAX(y, 
                order=(p, d, q),
                seasonal_order=(P, D, Q, S),
                enforce_stationarity=False, 
                enforce_invertibility=False,
                exog = exog)

# Fit the model
sarima_result = model.fit()

# Print summary
print(sarima_result.summary())


# Assuming the last row of the original data represents the most recent day
last_row = df_calender.iloc[-1]

# Repeat the last values of "holiday_flag" and "tcb_holiday_flag" for the next 14 days
forecast_exog = pd.DataFrame({
    "holiday_flag": [last_row["holiday_flag"]] * 14,
    "tcb_holiday_flag": [last_row["tcb_holiday_flag"]] * 14
}, index=pd.date_range(start=df_calender.index[-1] + pd.Timedelta(days=1), periods=14, freq='D'))

# Now, get the forecast with the exogenous variables for the next 14 days
forecast = sarima_result.get_forecast(steps=14, exog=forecast_exog)

# Get confidence intervals
forecast_index = pd.date_range(start=y.index[-1]+ pd.Timedelta(days=1), periods=14, freq="D")
forecast_mean = forecast.predicted_mean.astype(int)  # 小数点を整数に変換
forecast_ci = forecast.conf_int()
forecast_top = forecast_ci.iloc[:, 1].astype(int)  # 上限
forecast_bot = forecast_ci.iloc[:, 0].astype(int)  # 下限

# Plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(y, label="Actual", color="blue")
plt.plot(forecast_index, forecast_mean, label="Forecast", color="red")
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color="pink", alpha=0.3)
plt.xlabel("Date")
plt.ylabel("Counseled")
plt.title("SARIMA Forecast for Counseled")
plt.legend()
plt.grid()
plt.show()

# In-sample predictions
y_pred = sarima_result.predict(start=0, end=len(y)-1)

forecast_mean = forecast_mean.apply(lambda x: int(x)) 

# **予測データを CSV に保存**
forecast_df = pd.DataFrame({
    "Date": forecast_index,
    "Forecast": forecast_mean,
    "Forecast 95% Top": forecast_top,
    "Forecast 95% Bot": forecast_bot
})
forecast_df.to_csv("../data/great_ver4.csv", index=False)

# Calculate errors
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

