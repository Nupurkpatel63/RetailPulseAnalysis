
from utils.lstm_predictor import (
    load_lstm_model,
    load_revenue_scaler,
    load_revenue_data,
    forecast_revenue,
)

try:
    series = load_revenue_data()
    scaler = load_revenue_scaler()
    model = load_lstm_model()

    print("Daily revenue data loaded:", len(series), "rows")
    print("Scaler loaded:", type(scaler).__name__)
    print("LSTM model loaded:", type(model).__name__)

    forecast = forecast_revenue(days=7)
    print("\nNext 7 days forecast:")
    print(forecast)

except Exception as e:
    print("Error:", e)
