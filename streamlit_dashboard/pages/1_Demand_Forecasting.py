
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_loader import render_csv_uploader

from utils.lstm_predictor import (
    load_revenue_data,
    forecast_revenue,
    retrain_lstm,
)

st.set_page_config(
    page_title="Demand Forecasting | RetailPulse",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Demand Forecasting")
st.markdown(
    "Forecast future revenue using your trained LSTM model "
    "and explore possible changes with what-if analysis."
)

render_csv_uploader()

st.divider()

# Sidebar controls
st.sidebar.header("Forecast Settings")

forecast_days = st.sidebar.selectbox(
    "Forecast horizon",
    options=[7, 14, 30],
    index=0,
)

what_if_change = st.sidebar.slider(
    "What-if revenue change (%)",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
)

history_days = st.sidebar.selectbox(
    "Historical days to display",
    options=[30, 60, 90, 180],
    index=1,
)

st.subheader("Retrain Revenue Model")
st.write(
    "Train the LSTM on the active dataset. The current cleaned dataset "
    "is used until you upload a CSV. At least 38 daily observations are needed."
)
st.caption(
    "Active dataset: "
    + st.session_state.get(
        "retailpulse_data_source_name", "Current cleaned dataset"
    )
)

if st.button("Retrain LSTM with active dataset", type="primary"):
    try:
        with st.spinner("Training LSTM and updating the saved model..."):
            active_history = load_revenue_data()
            training_result = retrain_lstm(active_history)
        st.success(
            "LSTM retrained and saved. "
            f"Best validation loss: {training_result['validation_loss']:.6f} "
            f"({training_result['epochs']} epochs)."
        )
    except Exception as exc:
        st.error(f"LSTM retraining failed: {exc}")


# Load data and forecast
try:
    with st.spinner("Generating revenue forecast..."):
        historical_revenue = load_revenue_data()
        forecast_df = forecast_revenue(
            days=forecast_days,
            series=historical_revenue,
        )

    # Apply the scenario adjustment to the forecast only
    forecast_df["Scenario_Revenue"] = (
        forecast_df["Forecast_Revenue"]
        * (1 + what_if_change / 100)
    )

    # Summary metrics
    total_forecast = forecast_df["Forecast_Revenue"].sum()
    scenario_total = forecast_df["Scenario_Revenue"].sum()
    average_forecast = forecast_df["Forecast_Revenue"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Forecast Revenue",
        f"{total_forecast:,.2f}",
        help="Total predicted revenue over the selected forecast period.",
    )

    col2.metric(
        "Average Daily Forecast",
        f"{average_forecast:,.2f}",
    )

    col3.metric(
        "What-if Scenario Total",
        f"{scenario_total:,.2f}",
        delta=f"{scenario_total - total_forecast:,.2f}",
        help="Forecast total after applying your percentage adjustment.",
    )

    st.divider()

    # Historical and forecast chart
    st.subheader("Historical Revenue vs Forecast")

    history = historical_revenue.tail(history_days)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Historical Revenue",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast_Revenue"],
            mode="lines+markers",
            name="LSTM Forecast",
            line=dict(dash="dash", width=3),
        )
    )

    if what_if_change != 0:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Scenario_Revenue"],
                mode="lines+markers",
                name=f"What-if Scenario ({what_if_change:+}%)",
                line=dict(width=2, dash="dot"),
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue",
        hovermode="x unified",
        legend_title="Series",
        template="plotly_white",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The what-if adjustment applies a percentage change to the forecast. "
        "Use Retrain Revenue Model above to train and save a model from the "
        "active dataset."
    )

    # Forecast detail table
    st.subheader("Forecast Details")

    display_df = forecast_df.copy()
    display_df["Date"] = pd.to_datetime(
    display_df["Date"]
    ).dt.strftime("%Y-%m-%d")

    display_df = display_df.rename(
        columns={
            "Date": "Date",
            "Forecast_Revenue": "Predicted Revenue",
            "Scenario_Revenue": "What-if Revenue",
        }
    )

    st.dataframe(
        display_df.round(2),
        use_container_width=True,
        hide_index=True,
    )

    # CSV download
    csv_data = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv_data,
        file_name="retailpulse_revenue_forecast.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error("Unable to generate the revenue forecast.")
    st.exception(e)
    st.info(
        "Verify the LSTM model, scaler, historical revenue data, "
        "required columns, and installed dependencies."
    )
