
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data
from utils.data_loader import load_data, render_csv_uploader

st.set_page_config(
    page_title="Sales Performance",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Sales Performance Dashboard")
st.markdown("Analyze revenue trends, products, and country performance.")

render_csv_uploader()
df = load_data()

# Ensure data types are correct
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")

df = df.dropna(subset=["InvoiceDate", "Revenue"]).copy()

# Optional date filter
min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

st.sidebar.subheader("Sales Filters")

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range

    filtered_df = df[
        (df["InvoiceDate"].dt.date >= start_date)
        & (df["InvoiceDate"].dt.date <= end_date)
    ].copy()
else:
    filtered_df = df.copy()

if filtered_df.empty:
    st.warning("No sales data available for the selected dates.")
    st.stop()

# KPI metrics
total_revenue = filtered_df["Revenue"].sum()
total_orders = filtered_df["Invoice"].nunique()
total_units = filtered_df["Quantity"].sum()
avg_order_value = (
    total_revenue / total_orders if total_orders else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"{total_revenue:,.2f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Net Units Sold", f"{total_units:,.0f}")
col4.metric("Average Order Value", f"{avg_order_value:,.2f}")

st.divider()

# Daily revenue trend
st.subheader("📈 Daily Revenue Trend")

daily_sales = (
    filtered_df.groupby(filtered_df["InvoiceDate"].dt.date)["Revenue"]
    .sum()
    .reset_index()
)

daily_sales.columns = ["Date", "Revenue"]

fig_daily = px.line(
    daily_sales,
    x="Date",
    y="Revenue",
    markers=True,
    title="Daily Revenue"
)

st.plotly_chart(fig_daily, use_container_width=True)

# Product and country analysis
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("🏆 Top 10 Products by Revenue")

    product_sales = (
        filtered_df.groupby("Description", dropna=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Units=("Quantity", "sum")
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )

    product_sales["Description"] = (
        product_sales["Description"].fillna("Unknown")
    )

    fig_products = px.bar(
        product_sales.sort_values("Revenue"),
        x="Revenue",
        y="Description",
        orientation="h",
        title="Top Products by Revenue"
    )

    st.plotly_chart(fig_products, use_container_width=True)

with right_col:
    st.subheader("🌍 Revenue by Country")

    country_sales = (
        filtered_df.groupby("Country")["Revenue"]
        .sum()
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )

    fig_country = px.bar(
        country_sales.sort_values("Revenue"),
        x="Revenue",
        y="Country",
        orientation="h",
        title="Top 10 Countries by Revenue"
    )

    st.plotly_chart(fig_country, use_container_width=True)

# Monthly revenue
st.subheader("📅 Monthly Revenue")

monthly_sales = filtered_df.copy()
monthly_sales["Month"] = (
    monthly_sales["InvoiceDate"].dt.to_period("M").astype(str)
)

monthly_sales = (
    monthly_sales.groupby("Month")["Revenue"]
    .sum()
    .reset_index()
)

fig_monthly = px.bar(
    monthly_sales,
    x="Month",
    y="Revenue",
    title="Monthly Revenue Trend"
)

st.plotly_chart(fig_monthly, use_container_width=True)

# Download filtered sales
st.subheader("⬇️ Export Sales Data")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Sales CSV",
    data=csv_data,
    file_name="sales_performance.csv",
    mime="text/csv"
)
