Understanding (p, d, q) and (P, D, Q, S) in SARIMA
SARIMA (Seasonal AutoRegressive Integrated Moving Average) extends ARIMA by incorporating seasonality. The parameters are:

Non-Seasonal Components (p, d, q)
These define the basic ARIMA model:

p (AutoRegressive order, AR): Number of lag observations included in the model.
d (Differencing order, I): Number of times the series needs to be differenced to make it stationary.
q (Moving Average order, MA): Number of lagged forecast errors included in the model.
Seasonal Components (P, D, Q, S)
These define the seasonal part of the SARIMA model:

P (Seasonal AutoRegressive order, SAR): Similar to p but for seasonal lags.
D (Seasonal Differencing order, SI): Number of times seasonal differencing is applied.
Q (Seasonal Moving Average order, SMA): Similar to q but for seasonal lags.
S (Seasonal period): The length of the seasonal cycle (e.g., S=12 for monthly data, S=7 for weekly data).
SARIMA Model Notation
A SARIMA model is typically written as:

SARIMA(p,d,q)×(P,D,Q,S)
For example, SARIMA(1,1,1)(1,1,1,12) would mean:

ARIMA(1,1,1) for the non-seasonal component
Seasonal ARIMA(1,1,1) with a periodicity of 12 (e.g., monthly data)
How to Choose (p, d, q) and (P, D, Q, S)
Determine d and D (Differencing Orders)

Check if the time series is stationary using an Augmented Dickey-Fuller (ADF) test.
If the ADF test fails to reject stationarity (p-value > 0.05), apply differencing (d or D).
Determine p and P (AR Terms)

Look at Partial Autocorrelation Function (PACF) plots:
If there’s a sharp cutoff at lag k, use p = k.
If no clear cutoff, try multiple values.
Determine q and Q (MA Terms)

Look at Autocorrelation Function (ACF) plots:
If there’s a sharp cutoff at lag k, use q = k.
Select S (Seasonal Periodicity)

Choose S based on known seasonality:
7 for daily data with weekly seasonality.
12 for monthly data.
24 for hourly data with daily seasonality.

<!-- Best SARIMA Parameters: {'p': 0, 'd': 0, 'q': 1, 'P': 2, 'D': 2, 'Q': 0, 'S': 100}
C:\Users\analyticsteam_share\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
C:\Users\analyticsteam_share\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
Best SARIMA AIC: 5076.450577587952 -->