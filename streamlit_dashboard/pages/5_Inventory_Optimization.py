
import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_data, render_csv_uploader


st.set_page_config(
    page_title="Inventory Optimization | RetailPulse",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Inventory Optimization")
st.caption(
    "Analyze historical product demand and generate "
    "inventory replenishment recommendations."
)


# --------------------------------------------------
# 1. Load and prepare data
# --------------------------------------------------

render_csv_uploader()

def get_inventory_data():

    df = load_data().copy()

    required_columns = [
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    demand = df.dropna(
        subset=["StockCode", "InvoiceDate", "Quantity"]
    ).copy()

    if "IsCancelled" in demand.columns:
        cancelled = (
            demand["IsCancelled"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        demand = demand.loc[~cancelled].copy()

    demand = demand.loc[
        demand["Quantity"] > 0
    ].copy()

    demand["Description"] = (
        demand["Description"]
        .fillna("Unknown product")
    )

    return demand


try:
    df = get_inventory_data()

except Exception as e:
    st.error(f"Unable to load inventory data: {e}")
    st.stop()


# --------------------------------------------------
# 2. Sidebar controls
# --------------------------------------------------
st.sidebar.header("Inventory Settings")

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

date_range = st.sidebar.date_input(
    "Demand analysis period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()

start_date, end_date = date_range

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

lead_time = st.sidebar.number_input(
    "Supplier lead time (days)",
    min_value=1,
    max_value=180,
    value=7
)

target_days = st.sidebar.number_input(
    "Target stock coverage (days)",
    min_value=1,
    max_value=365,
    value=30
)

safety_buffer = st.sidebar.slider(
    "Safety stock buffer (%)",
    min_value=0,
    max_value=100,
    value=20,
    step=5
)


# --------------------------------------------------
# 3. Filter selected period
# --------------------------------------------------
period_df = df.loc[
    df["InvoiceDate"].dt.date.between(
        start_date, end_date
    )
].copy()

if period_df.empty:
    st.warning("No sales found for the selected period.")
    st.stop()

analysis_days = (
    pd.Timestamp(end_date) - pd.Timestamp(start_date)
).days + 1


# --------------------------------------------------
# 4. Aggregate product demand
# --------------------------------------------------
product_summary = (
    period_df.groupby("StockCode", as_index=False)
    .agg(
        Product=("Description", "last"),
        Total_Units_Sold=("Quantity", "sum"),
        Sales_Lines=("InvoiceDate", "size"),
        Last_Sale_Date=("InvoiceDate", "max")
    )
)

product_summary["Avg_Daily_Demand"] = (
    product_summary["Total_Units_Sold"] / analysis_days
)

product_summary["Safety_Stock"] = (
    product_summary["Avg_Daily_Demand"]
    * lead_time
    * safety_buffer / 100
)

product_summary["Reorder_Point"] = (
    product_summary["Avg_Daily_Demand"] * lead_time
    + product_summary["Safety_Stock"]
).apply(np.ceil).astype(int)

product_summary["Target_Stock"] = (
    product_summary["Avg_Daily_Demand"] * target_days
    + product_summary["Safety_Stock"]
).apply(np.ceil).astype(int)


# --------------------------------------------------
# 5. Inventory dashboard metrics
# --------------------------------------------------
st.subheader("Inventory Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Products Analyzed",
    f"{product_summary['StockCode'].nunique():,}"
)

col2.metric(
    "Units Sold",
    f"{product_summary['Total_Units_Sold'].sum():,.0f}"
)

col3.metric(
    "Average Daily Demand",
    f"{product_summary['Avg_Daily_Demand'].sum():,.1f}"
)

col4.metric(
    "Analysis Days",
    f"{analysis_days:,}"
)

st.caption(
    f"Historical period: {start_date} to {end_date}. "
    "Recommendations are estimates based on observed positive sales."
)


# --------------------------------------------------
# 6. Current stock inputs
# --------------------------------------------------
st.subheader("Current Stock and Reorder Recommendations")

st.write(
    "Enter current stock for the products you want to assess. "
    "Stock levels are not available in the sales dataset, "
    "so they must be provided by you."
)

top_n = st.selectbox(
    "Number of products to display",
    options=[10, 20, 50, 100],
    index=1
)

product_summary = product_summary.sort_values(
    "Total_Units_Sold", ascending=False
).head(top_n).copy()

product_summary["Current_Stock"] = 0

edited_products = st.data_editor(
    product_summary[
        [
            "StockCode",
            "Product",
            "Total_Units_Sold",
            "Avg_Daily_Demand",
            "Safety_Stock",
            "Reorder_Point",
            "Target_Stock",
            "Current_Stock"
        ]
    ],
    column_config={
        "StockCode": st.column_config.TextColumn(
            "Product Code",
            disabled=True
        ),
        "Product": st.column_config.TextColumn(
            "Product",
            disabled=True
        ),
        "Total_Units_Sold": st.column_config.NumberColumn(
            "Units Sold",
            disabled=True,
            format="%.0f"
        ),
        "Avg_Daily_Demand": st.column_config.NumberColumn(
            "Avg Daily Demand",
            disabled=True,
            format="%.2f"
        ),
        "Safety_Stock": st.column_config.NumberColumn(
            "Safety Stock",
            disabled=True,
            format="%.0f"
        ),
        "Reorder_Point": st.column_config.NumberColumn(
            "Reorder Point",
            disabled=True,
            format="%.0f"
        ),
        "Target_Stock": st.column_config.NumberColumn(
            "Target Stock",
            disabled=True,
            format="%.0f"
        ),
        "Current_Stock": st.column_config.NumberColumn(
            "Current Stock",
            min_value=0,
            step=1,
            format="%d"
        )
    },
    hide_index=True,
    use_container_width=True,
    key="inventory_editor"
)


# --------------------------------------------------
# 7. Calculate recommendations
# --------------------------------------------------
recommendations = edited_products.copy()

recommendations["Current_Stock"] = pd.to_numeric(
    recommendations["Current_Stock"],
    errors="coerce"
).fillna(0).clip(lower=0)

recommendations["Recommended_Order_Qty"] = (
    recommendations["Target_Stock"]
    - recommendations["Current_Stock"]
).clip(lower=0).apply(np.ceil).astype(int)

recommendations["Status"] = np.where(
    recommendations["Current_Stock"]
    <= recommendations["Reorder_Point"],
    "Reorder",
    "Stock Sufficient"
)

reorder_count = (
    recommendations["Status"] == "Reorder"
).sum()

st.subheader("Recommendation Summary")

m1, m2, m3 = st.columns(3)

m1.metric("Products Needing Reorder", f"{reorder_count:,}")
m2.metric(
    "Total Suggested Units",
    f"{recommendations['Recommended_Order_Qty'].sum():,.0f}"
)
m3.metric(
    "Products With Sufficient Stock",
    f"{(recommendations['Status'] == 'Stock Sufficient').sum():,}"
)

st.dataframe(
    recommendations.sort_values(
        ["Status", "Recommended_Order_Qty"],
        ascending=[True, False]
    ),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 8. Chart: highest-demand products
# --------------------------------------------------
st.subheader("Top Products by Historical Demand")

chart_df = recommendations.nlargest(
    10, "Total_Units_Sold"
).sort_values("Total_Units_Sold")

st.bar_chart(
    chart_df.set_index("Product")["Total_Units_Sold"]
)


# --------------------------------------------------
# 9. Download report
# --------------------------------------------------
st.subheader("Export Inventory Report")

csv_data = recommendations.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Inventory Recommendations (CSV)",
    data=csv_data,
    file_name="retailpulse_inventory_recommendations.csv",
    mime="text/csv"
)

st.info(
    "Planning note: current stock is user-entered. "
    "The reorder point uses average daily demand × lead time "
    "plus a configurable safety buffer. It is a simple planning "
    "estimate, not a supplier purchase order or a live inventory feed."
)
