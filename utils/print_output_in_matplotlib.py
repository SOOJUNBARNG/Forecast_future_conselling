import pandas as pd
import matplotlib.pyplot as plt

def plot_result(y, forecast, clinic_id):

    # Get confidence intervals
    forecast_index = pd.date_range(
        start=y.index[-1] + pd.Timedelta(days=1), periods=28, freq="D"
    )
    forecast_mean = forecast.predicted_mean.astype(int)
    forecast_ci = forecast.conf_int()
    forecast_top = forecast_ci.iloc[:, 1].astype(int)
    forecast_bot = forecast_ci.iloc[:, 0].astype(int)

    plt.figure(figsize=(12, 6))
    plt.plot(y, label="Actual", color="blue")
    plt.plot(forecast_index, forecast_mean, label="Forecast", color="red")
    plt.fill_between(
        forecast_index, forecast_bot, forecast_top, color="pink", alpha=0.3
    )
    plt.xlabel("Date")
    plt.ylabel("Counseled")
    plt.title(f"SARIMA Forecast for Clinic {clinic_id}")
    plt.legend()
    plt.grid()
    plt.show()
