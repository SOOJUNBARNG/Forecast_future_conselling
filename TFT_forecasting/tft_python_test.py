# 必要なライブラリのインポート
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

# データセットの定義
class CryptoDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index+self.seq_length]
        y = self.data[index+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Temporal Fusion Transformerの簡易モデル定義
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(TemporalFusionTransformer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ハイパーパラメータの設定
input_size = 1
hidden_size = 64
output_size = 30
num_layers = 2
seq_length = 30
num_epochs = 100
learning_rate = 0.001

# データの読み込みと前処理
data = pd.read_csv('../data/counseling_count_group.csv')['counseled'].values
# print(data.columns)

# Get the last `seq_length` values for prediction input
test_input = data[-seq_length:]

# Convert to tensor and reshape for LSTM input
test_input = torch.tensor(test_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # Shape: (1, seq_length, 1)

# モデルの初期化
model = TemporalFusionTransformer(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Run inference
with torch.no_grad():
    prediction = model(test_input)  # Shape: (1, output_size)

# Convert tensor to list
predicted_values = prediction.squeeze().tolist()  # Convert to a flat list

# Create DataFrame
df = pd.DataFrame({"Predicted Value": predicted_values})

# Save to CSV
df.to_csv("prediction_results.csv", index=False, encoding="utf-8-sig")

print("Predictions saved to prediction_results.csv")