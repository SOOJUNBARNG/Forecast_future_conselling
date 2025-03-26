import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

# Load the Chronos pipeline
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# Load and preprocess the data
df = pd.read_csv("../data/counseling_count.csv")
df = df.groupby("日付")["counseled"].sum()  # No reset_index()
df = df.reset_index()  # Reset index
df["日付"] = pd.to_datetime(df["日付"])  # Convert to datetime
df = df[df["日付"] <= "2025-01-31"]  # Filter by date

# Prepare the context as a tensor
context = torch.tensor(df.values, dtype=torch.float32)

# Set prediction length
prediction_length = 28

# Get the forecast
forecast = pipeline.predict(
    context, prediction_length
)  # shape [num_series, num_samples, prediction_length]

# Visualize the forecast
forecast_index = range(len(df), len(df) + prediction_length)

# Ensure tensor is moved to CPU if it's on CUDA
low, median, high = np.quantile(forecast[0].cpu().numpy(), [0.1, 0.5, 0.9], axis=0)

plt.figure(figsize=(8, 4))
plt.plot(
    df.index, df.values, color="royalblue", label="historical data"
)  # Use the index without resetting
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(
    forecast_index,
    low,
    high,
    color="tomato",
    alpha=0.3,
    label="80% prediction interval",
)
plt.legend()
plt.grid()
plt.show()

# Output forecast to CSV
forecast_df = pd.DataFrame(
    {
        "forecast_index": forecast_index,
        "low": low,
        "median": median,
        "high": high,
    }
)

forecast_df.to_csv("forecast_output.csv", index=False)

# plt.figure(figsize=(8, 4))
# plt.plot(df["counseled"], color="royalblue", label="historical data")
# plt.plot(forecast_index, median, color="tomato", label="median forecast")
# plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
# plt.legend()
# plt.grid()
# plt.show()
