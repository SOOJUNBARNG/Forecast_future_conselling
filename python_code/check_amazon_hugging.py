import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
  "amazon/chronos-t5-large",
  device_map="cuda",
  torch_dtype=torch.bfloat16,
)

df = pd.read_csv("D:\Forecast_future_conselling\data\check_df.csv")

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
df["date"] = pd.to_datetime(df["日付"])
context = torch.tensor(df["counseled"])
prediction_length = 12
forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]
forecast_dates = pd.date_range(start=df["date"].iloc[-1], periods=prediction_length + 1, freq="D")[1:]

# visualize the forecast
forecast_index = range(len(df), len(df) + prediction_length)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

# Plot with correct date format
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["counseled"], color="royalblue", label="Historical Data")
plt.plot(forecast_dates, median, color="tomato", label="Median Forecast")
plt.fill_between(forecast_dates, low, high, color="tomato", alpha=0.3, label="80% Prediction Interval")

# Add forecast details as text annotations
for i, (d, m) in enumerate(zip(forecast_dates, median)):
    plt.text(d, m, f"{m:.1f}", ha="center", fontsize=8, color="black")

# Formatting
plt.xlabel("日付")
plt.ylabel("予測値")
plt.title("Counseled Forecast")
plt.xticks(rotation=45)  # Rotate dates for better visibility
plt.legend()
plt.grid()
plt.show()