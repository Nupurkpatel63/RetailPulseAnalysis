
from pathlib import Path
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import lightning.pytorch as pl
from utils.data_loader import load_data


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "model" / "lstm_revenue_model.pth"
SCALER_PATH = PROJECT_DIR / "model" / "revenue_scaler.pkl"

LOOKBACK = 28
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RevenueLSTM(pl.LightningModule):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.fc(last_output).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )


def load_revenue_data():
    df = load_data()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    daily = df.groupby(
        df["InvoiceDate"].dt.normalize()
    )["Revenue"].sum().sort_index()

    if daily.empty:
        raise ValueError("The active dataset has no daily revenue records.")

    # Treat dates without transactions as zero-sales days for the model.
    daily.index = pd.DatetimeIndex(daily.index)
    daily = daily.asfreq("D", fill_value=0).astype(float)
    daily.index.name = "Date"
    daily.name = "Revenue"
    return daily


def load_revenue_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Revenue scaler not found: {SCALER_PATH}"
        )

    return joblib.load(SCALER_PATH)


def retrain_lstm(series=None, epochs=50, patience=7):
    """Train the revenue LSTM on the active dataset and save its artifacts."""

    if series is None:
        series = load_revenue_data()

    values = np.asarray(series, dtype=np.float32).reshape(-1, 1)
    if len(values) < LOOKBACK + 10:
        raise ValueError(
            f"At least {LOOKBACK + 10} daily observations are required "
            "to retrain the LSTM."
        )

    scaler = MinMaxScaler()
    scaler_fit_end = max(LOOKBACK + 1, int(len(values) * 0.8))
    scaler.fit(values[:scaler_fit_end])
    scaled = scaler.transform(values).astype(np.float32)

    windows = np.asarray([
        scaled[index - LOOKBACK:index]
        for index in range(LOOKBACK, len(scaled))
    ], dtype=np.float32)
    targets = scaled[LOOKBACK:, 0]

    split_at = max(1, int(len(windows) * 0.8))
    if split_at >= len(windows):
        split_at = len(windows) - 1

    x_train = torch.from_numpy(windows[:split_at]).to(DEVICE)
    y_train = torch.from_numpy(targets[:split_at]).to(DEVICE)
    x_valid = torch.from_numpy(windows[split_at:]).to(DEVICE)
    y_valid = torch.from_numpy(targets[split_at:]).to(DEVICE)

    model = RevenueLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model.hparams.learning_rate
    )
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    completed_epochs = 0

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_train), device=DEVICE)
        for batch_indices in permutation.split(64):
            optimizer.zero_grad()
            predictions = model(x_train[batch_indices])
            loss = model.loss_fn(predictions, y_train[batch_indices])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = model.loss_fn(model(x_valid), y_valid).item()

        completed_epochs = epoch + 1
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = completed_epochs
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_epoch == 0:
        raise RuntimeError("LSTM training did not produce a valid checkpoint.")

    # Use the validation-selected training duration, then fit the deployable
    # model and scaler on the full uploaded history.
    final_scaler = MinMaxScaler().fit(values)
    final_scaled = final_scaler.transform(values).astype(np.float32)
    final_windows = torch.from_numpy(np.asarray([
        final_scaled[index - LOOKBACK:index]
        for index in range(LOOKBACK, len(final_scaled))
    ], dtype=np.float32)).to(DEVICE)
    final_targets = torch.from_numpy(
        final_scaled[LOOKBACK:, 0]
    ).to(DEVICE)

    final_model = RevenueLSTM().to(DEVICE)
    final_optimizer = torch.optim.Adam(
        final_model.parameters(), lr=final_model.hparams.learning_rate
    )
    for _ in range(best_epoch):
        final_model.train()
        permutation = torch.randperm(len(final_windows), device=DEVICE)
        for batch_indices in permutation.split(64):
            final_optimizer.zero_grad()
            predictions = final_model(final_windows[batch_indices])
            loss = final_model.loss_fn(
                predictions, final_targets[batch_indices]
            )
            loss.backward()
            final_optimizer.step()

    # Write temporary artifacts first so failed training never replaces the
    # model currently used by the dashboard.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_temp = None
    scaler_temp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=MODEL_PATH.parent, suffix=".pth", delete=False
        ) as temp_file:
            model_temp = Path(temp_file.name)
        with tempfile.NamedTemporaryFile(
            dir=SCALER_PATH.parent, suffix=".pkl", delete=False
        ) as temp_file:
            scaler_temp = Path(temp_file.name)

        torch.save(final_model.state_dict(), model_temp)
        joblib.dump(final_scaler, scaler_temp)
        os.replace(scaler_temp, SCALER_PATH)
        os.replace(model_temp, MODEL_PATH)
    finally:
        for temp_path in (model_temp, scaler_temp):
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    return {
        "epochs": best_epoch,
        "validation_loss": best_loss,
        "training_observations": len(x_train),
        "validation_observations": len(x_valid),
    }


def load_lstm_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LSTM model not found: {MODEL_PATH}"
        )

    model = RevenueLSTM()

    # Assumes the file contains model.state_dict().
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def forecast_revenue(days=7, series=None):
    if days < 1:
        raise ValueError("Forecast days must be at least 1.")

    if series is None:
        series = load_revenue_data()
    scaler = load_revenue_scaler()
    model = load_lstm_model()

    if len(series) < LOOKBACK:
        raise ValueError(
            f"At least {LOOKBACK} days of revenue are required."
        )

    scaled = scaler.transform(
        series.to_numpy().reshape(-1, 1)
    )

    window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    forecasts = []

    with torch.no_grad():
        for _ in range(days):
            x = torch.tensor(
                window,
                dtype=torch.float32,
                device=DEVICE,
            )

            prediction = model(x).cpu().numpy().reshape(-1, 1)
            forecasts.append(float(prediction[0, 0]))

            # Use the prediction as the next input day.
            window = np.concatenate(
                [window[:, 1:, :], prediction.reshape(1, 1, 1)],
                axis=1,
            )

    predictions = scaler.inverse_transform(
        np.array(forecasts).reshape(-1, 1)
    ).ravel()

    future_dates = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=days,
        freq="D",
    )

    return pd.DataFrame({
        "Date": future_dates,
        "Forecast_Revenue": predictions,
    })
