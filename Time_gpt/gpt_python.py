import pandas as pd
import requests

# Load data
data = pd.read_csv("../data/prepare_forecast.csv", parse_dates=["date"])

# Prepare data for TimeGPT API
payload = {
    "time": data["date"].astype(str).tolist(),
    "y": data["counseled"].tolist(),
    "horizon": 7  # Predict next 7 days
}

# Define API endpoint (update with actual API key and endpoint if needed)
API_KEY = "nixak-y9A06fB6IzHCYbTf3m8ztnczqExB7qI9MREiuVvtJfuWUNF1YBRwJRcd0vBDRobJ3D9SmvAQ9S8yJtZo"
API_URL = "https://timegpt.nixtla.io/forecast"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Send request
response = requests.post(API_URL, json=payload, headers=headers)

# Process response
if response.status_code == 200:
    predictions = response.json()
    print("Predictions:", predictions)
else:
    print("Error:", response.text)
