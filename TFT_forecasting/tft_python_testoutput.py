import torch
import numpy as np
from your_tft_model import TemporalFusionTransformer  # Replace with your actual model class

# Define model structure (must match the trained model)
model = TemporalFusionTransformer(
    input_size=..., 
    hidden_size=..., 
    output_size=..., 
    num_lstm_layers=..., 
    num_attention_heads=...  
)  # Ensure parameters match training

# Load saved weights
model.load_state_dict(torch.load('tft_model.pth'))
model.eval()  # Set model to evaluation mode

# Example input (replace with real test data)
test_input = torch.tensor(np.random.rand(1, time_steps, input_size), dtype=torch.float32)  

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
test_input = test_input.to(device)
