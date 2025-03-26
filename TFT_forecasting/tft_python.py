import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_lightning import Trainer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping

# Load data
data = pd.read_csv("../data/prepare_forecast.csv", parse_dates=["date"])

# Define max prediction and context length
max_prediction_length = 7  # Forecast 7 days ahead
max_encoder_length = 30     # Use past 30 days for context

data["time_idx"] = (data["date"] - data["date"].min()).dt.days

target = "counseled"
group_ids = ["clinic_holiday"]  # Define groups for the model

data["day"] = data["day"].astype(str)
data["week_of_month"] = data["week_of_month"].astype(str)
data["national_holiday"] = data["national_holiday"].astype(str)
data["clinic_holiday"] = data["clinic_holiday"].astype(str)

# Create dataset
tft_dataset = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target=target,
    group_ids=group_ids,
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    time_varying_known_categoricals=["day", "week_of_month", "national_holiday", "clinic_holiday"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["counseled"],
    target_normalizer=GroupNormalizer(groups=group_ids),
    allow_missing_timesteps=True
)

# Create dataloaders
train_dataloader = tft_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
val_dataloader = tft_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

# Define model
tft = TemporalFusionTransformer.from_dataset(
    tft_dataset,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=10,
    log_val_interval=1,
)

# Train model
trainer = Trainer(
    max_epochs=30,
    devices=1,  # Use a single GPU
    gradient_clip_val=0.1,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")]
)

trainer.fit(tft, train_dataloader, val_dataloader)

# Predict
raw_predictions, x = tft.predict(val_dataloader, mode="raw", return_x=True)
print("Predictions:", raw_predictions)
