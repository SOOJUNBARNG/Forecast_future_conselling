from prophet import Prophet
df = pd.DataFrame({'ds': dates, 'y': counseling_counts})
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
