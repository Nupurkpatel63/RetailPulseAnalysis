
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data, render_csv_uploader

from utils.data_loader import load_data

st.set_page_config(
    page_title="Customer Analytics | RetailPulse",
    page_icon="👥",
    layout="wide",
)

st.title("👥 Customer Analytics")
st.write(
    "Analyze customer purchasing behavior, identify valuable "
    "customers, and explore RFM-based customer segments."
)

render_csv_uploader()

@st.cache_data
def build_customer_metrics(df):
    required = {
        "Customer ID", "Invoice", "InvoiceDate", "Revenue"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    data = df.copy()
    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )
    data["Customer ID"] = data["Customer ID"].astype("string").str.strip()
    data["Revenue"] = pd.to_numeric(
        data["Revenue"], errors="coerce"
    )

    data = data.dropna(
        subset=["Customer ID", "Invoice", "InvoiceDate", "Revenue"]
    )

    # Exclude cancellation rows if the dataset identifies them.
    # Confirm IsCancelled semantics in your cleaning notebook.
    if "IsCancelled" in data.columns:
        cancelled = data["IsCancelled"].astype("string").str.lower().isin(
            ["true", "1", "yes"]
        )
        data = data.loc[~cancelled].copy()

    if data.empty:
        raise ValueError("No usable customer transactions were found.")

    reference_date = data["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    rfm = data.groupby("Customer ID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
    )

    rfm["Recency"] = (
        reference_date - rfm["LastPurchase"].dt.normalize()
    ).dt.days

    # Rank-based scoring works even when many customers share values.
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first", ascending=True),
        4,
        labels=[4, 3, 2, 1],
    ).astype(int)

    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        4,
        labels=[1, 2, 3, 4],
    ).astype(int)

    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        4,
        labels=[1, 2, 3, 4],
    ).astype(int)

    def assign_segment(row):
        if row["Monetary"] <= 0:
            return "Non-positive Net Spend"
        if row["R_Score"] >= 3 and row["F_Score"] >= 3:
            return "Champions"
        if row["R_Score"] >= 2 and row["F_Score"] >= 3:
            return "Loyal"
        return "Other"

    rfm["Segment"] = rfm.apply(assign_segment, axis=1)
    return rfm.reset_index()


try:
    df = load_data()
    customers = build_customer_metrics(df)

    st.sidebar.header("Customer Filters")

    segments = sorted(customers["Segment"].unique().tolist())
    selected_segments = st.sidebar.multiselect(
        "Select customer segments",
        options=segments,
        default=segments,
    )

    min_frequency = int(customers["Frequency"].min())
    max_frequency = int(customers["Frequency"].max())

    frequency_range = st.sidebar.slider(
        "Number of distinct invoices",
        min_value=min_frequency,
        max_value=max_frequency,
        value=(min_frequency, max_frequency),
    )

    filtered = customers[
        customers["Segment"].isin(selected_segments)
        & customers["Frequency"].between(*frequency_range)
    ].copy()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers", f"{len(filtered):,}")
    col2.metric(
        "Average Net Spend",
        f"{filtered['Monetary'].mean():,.2f}"
        if not filtered.empty else "—",
    )
    col3.metric(
        "Average Invoices",
        f"{filtered['Frequency'].mean():,.1f}"
        if not filtered.empty else "—",
    )
    col4.metric(
        "Average Recency (days)",
        f"{filtered['Recency'].mean():,.1f}"
        if not filtered.empty else "—",
    )

    st.divider()

    if filtered.empty:
        st.warning("No customers match the selected filters.")
    else:
        left, right = st.columns(2)

        with left:
            st.subheader("Customer Segment Distribution")
            segment_counts = (
                filtered["Segment"].value_counts()
                .rename_axis("Segment")
                .reset_index(name="Customers")
            )
            fig_segments = px.bar(
                segment_counts,
                x="Segment",
                y="Customers",
                color="Segment",
                text="Customers",
            )
            fig_segments.update_layout(
                showlegend=False,
                template="plotly_white",
            )
            st.plotly_chart(fig_segments, use_container_width=True)

        with right:
            st.subheader("Frequency vs Net Spend")
            fig_scatter = px.scatter(
                filtered,
                x="Frequency",
                y="Monetary",
                color="Segment",
                hover_data=["Customer ID", "Recency"],
                labels={
                    "Frequency": "Distinct Invoices",
                    "Monetary": "Net Spend",
                },
            )
            fig_scatter.update_layout(template="plotly_white")
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Customer Details")

        search_id = st.text_input("Search Customer ID")
        table = filtered.copy()

        if search_id.strip():
            table = table[
                table["Customer ID"].str.contains(
                    search_id.strip(), case=False, na=False
                )
            ]

        table = table.sort_values("Monetary", ascending=False)

        st.dataframe(
            table[
                [
                    "Customer ID", "Recency", "Frequency",
                    "Monetary", "R_Score", "F_Score",
                    "M_Score", "Segment",
                ]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Customer Analytics CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_customer_analytics.csv",
            mime="text/csv",
        )

    st.caption(
        "RFM segments are rule-based summaries of historical transactions. "
        "They are not churn predictions."
    )

except Exception as e:
    st.error("Unable to load customer analytics.")
    st.exception(e)
