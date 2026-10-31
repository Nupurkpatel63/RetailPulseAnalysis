
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.lstm_predictor import RevenueLSTM, LOOKBACK

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR.parent / "data" / "daily_revenue.csv"
MODEL_PATH = PROJECT_DIR.parent / "model" / "lstm_revenue_model.pth"
SCALER_PATH = PROJECT_DIR.parent / "model" / "revenue_scaler.pkl"

OUTPUT_PATH = PROJECT_DIR / "data" / "forecast_evaluation.csv"

# Load daily revenue
df = pd.read_csv(DATA_PATH)

# Identify date and revenue columns
date_col = next(
    (c for c in df.columns if c.lower() in ["date", "ds", "invoicedate"]),
    None
)
revenue_col = next(
    (c for c in df.columns if c.lower() in
     ["revenue", "daily_revenue", "y"]),
    None
)

if date_col is None or revenue_col is None:
    raise ValueError(
        f"Could not identify date/revenue columns. Found: {df.columns.tolist()}"
    )

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[revenue_col] = pd.to_numeric(df[revenue_col], errors="coerce")

df = (
    df.dropna(subset=[date_col, revenue_col])
    .sort_values(date_col)
    .drop_duplicates(subset=[date_col], keep="last")
    .reset_index(drop=True)
)

if len(df) <= LOOKBACK + 7:
    raise ValueError("Not enough daily observations for evaluation.")

# Reserve the final 7 days for evaluation
test_size = 7
train_values = df[revenue_col].iloc[:-test_size].to_numpy()
actual_values = df[revenue_col].iloc[-test_size:].to_numpy()

# IMPORTANT:
# Fit a separate scaler on the earlier data to avoid using test values.
# The model's saved scaler is used for the model's expected input scaling.
# This evaluation assumes the saved model was trained with compatible scaling.
scaler = joblib.load(SCALER_PATH)
scaled_all = scaler.transform(
    df[[revenue_col]].to_numpy()
).reshape(-1)

# Create rolling one-step-ahead predictions.
# Each prediction uses the preceding 28 actual observations.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RevenueLSTM(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    learning_rate=0.001
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

predictions = []

with torch.no_grad():
    for idx in range(len(df) - test_size, len(df)):
        sequence = scaled_all[idx - LOOKBACK:idx]
        x = torch.tensor(
            sequence,
            dtype=torch.float32
        ).reshape(1, LOOKBACK, 1).to(device)

        prediction_scaled = model(x).detach().cpu().numpy().reshape(-1, 1)
        prediction = scaler.inverse_transform(prediction_scaled)[0, 0]
        predictions.append(float(prediction))

result = pd.DataFrame({
    "Date": df[date_col].iloc[-test_size:].to_numpy(),
    "Actual_Revenue": actual_values,
    "Predicted_Revenue": predictions
})

mae = mean_absolute_error(
    result["Actual_Revenue"], result["Predicted_Revenue"]
)
rmse = np.sqrt(mean_squared_error(
    result["Actual_Revenue"], result["Predicted_Revenue"]
))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)

print("Evaluation file saved:", OUTPUT_PATH)
print(result.to_string(index=False))
print(f"\nMAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")

