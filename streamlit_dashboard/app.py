
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
from utils.data_loader import load_data, render_csv_uploader

from utils.data_loader import load_data

st.set_page_config(
    page_title="RetailPulse | Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 RetailPulse")
st.subheader("AI-Driven Retail Analytics & Revenue Forecasting")

st.markdown(
    """
    Welcome to RetailPulse, an interactive dashboard for understanding
    retail performance, customer behavior, and revenue forecasts.
    Use the sidebar to explore each analytical module.
    """
)

# Load data

render_csv_uploader()

df = load_data()

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)
df["Revenue"] = pd.to_numeric(
    df["Revenue"], errors="coerce"
)

df = df.dropna(subset=["InvoiceDate", "Revenue"]).copy()

if df.empty:
    st.warning("No valid sales records are available.")
    st.stop()

# Summary metrics
total_revenue = df["Revenue"].sum()
total_orders = df["Invoice"].nunique()
total_customers = df["Customer ID"].nunique()
total_products = df["StockCode"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Net Revenue", f"{total_revenue:,.2f}")
col2.metric("Orders", f"{total_orders:,}")
col3.metric("Customers", f"{total_customers:,}")
col4.metric("Unique Products", f"{total_products:,}")

st.divider()

# Daily revenue chart
st.subheader("📈 Revenue Overview")

daily_revenue = (
    df.groupby(df["InvoiceDate"].dt.normalize())["Revenue"]
    .sum()
    .reset_index()
)
daily_revenue.columns = ["Date", "Revenue"]

fig = px.line(
    daily_revenue,
    x="Date",
    y="Revenue",
    title="Daily Net Revenue",
    markers=True
)
st.plotly_chart(fig, use_container_width=True)

# Latest data information
st.subheader("📌 Dataset Information")

left, right = st.columns(2)

with left:
    st.write("**First transaction date:**",
             df["InvoiceDate"].min().strftime("%d %b %Y"))
    st.write("**Last transaction date:**",
             df["InvoiceDate"].max().strftime("%d %b %Y"))

with right:
    st.write("**Countries represented:**",
             df["Country"].nunique())
    st.write("**Records analyzed:**",
             f"{len(df):,}")

st.info(
    "Use the sidebar pages for Demand Forecasting, Customer Analytics, "
    "Sales Performance, and Model Performance."
)

st.caption(
    "Note: Revenue metrics represent net values unless returns and "
    "cancellations are explicitly excluded."
)
