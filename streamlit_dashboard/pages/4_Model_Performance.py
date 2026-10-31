
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.data_loader import load_data, render_csv_uploader

from utils.data_loader import load_data

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance & Forecast Evaluation")
st.markdown(
    "Evaluate revenue forecasts using actual revenue, predicted revenue, "
    "MAE, and RMSE."
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

# Load historical data
render_csv_uploader()
df = load_data()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")

df = df.dropna(subset=["InvoiceDate", "Revenue"]).copy()

# Aggregate revenue by day
daily_actual = (
    df.groupby(df["InvoiceDate"].dt.normalize())["Revenue"]
    .sum()
    .sort_index()
    .rename("Actual_Revenue")
    .reset_index()
)

daily_actual.columns = ["Date", "Actual_Revenue"]

st.subheader("1. Historical Revenue")

fig_history = px.line(
    daily_actual,
    x="Date",
    y="Actual_Revenue",
    title="Daily Actual Revenue"
)
st.plotly_chart(fig_history, use_container_width=True)

# Try to load a saved forecast/actual comparison
forecast_candidates = [
    DATA_DIR / "forecast_evaluation.csv",
    DATA_DIR / "actual_vs_forecast.csv",
    DATA_DIR / "lstm_predictions.csv",
]

forecast_df = None
forecast_path = None

for path in forecast_candidates:
    if path.exists():
        candidate = pd.read_csv(path)

        # Identify date, actual and prediction columns
        lower_cols = {str(c).lower(): c for c in candidate.columns}

        date_col = next(
            (lower_cols[c] for c in ["date", "ds", "invoicedate"]
             if c in lower_cols),
            None
        )
        actual_col = next(
            (lower_cols[c] for c in
             ["actual_revenue", "actual", "y", "revenue"]
             if c in lower_cols),
            None
        )
        pred_col = next(
            (lower_cols[c] for c in
             ["forecast_revenue", "predicted_revenue", "prediction",
              "predicted", "yhat"]
             if c in lower_cols),
            None
        )

        if date_col and actual_col and pred_col:
            candidate = candidate.rename(columns={
                date_col: "Date",
                actual_col: "Actual_Revenue",
                pred_col: "Predicted_Revenue"
            })

            candidate["Date"] = pd.to_datetime(
                candidate["Date"], errors="coerce"
            )
            candidate["Actual_Revenue"] = pd.to_numeric(
                candidate["Actual_Revenue"], errors="coerce"
            )
            candidate["Predicted_Revenue"] = pd.to_numeric(
                candidate["Predicted_Revenue"], errors="coerce"
            )

            candidate = candidate.dropna(
                subset=["Date", "Actual_Revenue", "Predicted_Revenue"]
            )

            if not candidate.empty:
                forecast_df = candidate
                forecast_path = path
                break

if forecast_df is not None:
    st.success(f"Loaded evaluation data from: {forecast_path.name}")

    # Optional evaluation date filter
    forecast_df = forecast_df.sort_values("Date")

    st.subheader("2. Actual vs Predicted Revenue")

    fig_compare = px.line(
        forecast_df,
        x="Date",
        y=["Actual_Revenue", "Predicted_Revenue"],
        markers=True,
        title="Actual vs Predicted Revenue"
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # Metrics
    y_true = forecast_df["Actual_Revenue"]
    y_pred = forecast_df["Predicted_Revenue"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("Evaluated Observations", f"{len(forecast_df):,}")

    # Error analysis
    forecast_df["Forecast_Error"] = (
        forecast_df["Actual_Revenue"] -
        forecast_df["Predicted_Revenue"]
    )

    st.subheader("3. Forecast Error Analysis")

    fig_error = px.bar(
        forecast_df,
        x="Date",
        y="Forecast_Error",
        title="Forecast Error (Actual − Predicted)"
    )
    st.plotly_chart(fig_error, use_container_width=True)

    st.caption(
        "Positive error means the model underpredicted revenue; "
        "negative error means it overpredicted revenue."
    )

    st.download_button(
        "Download Evaluation Results",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="forecast_evaluation_results.csv",
        mime="text/csv"
    )

else:
    st.warning(
        "No saved actual-versus-predicted evaluation file was found. "
        "The historical chart is available, but MAE and RMSE cannot be "
        "calculated until predictions are compared with actual values."
    )

    st.info(
        "Expected file examples: data/forecast_evaluation.csv or "
        "data/actual_vs_forecast.csv. The file must contain a date column, "
        "an actual revenue column, and a predicted revenue column."
    )

    st.subheader("Evaluation file format")

    example = pd.DataFrame({
        "Date": ["2010-12-01", "2010-12-02"],
        "Actual_Revenue": [1000, 1200],
        "Predicted_Revenue": [950, 1250]
    })

    st.dataframe(example, use_container_width=True)
