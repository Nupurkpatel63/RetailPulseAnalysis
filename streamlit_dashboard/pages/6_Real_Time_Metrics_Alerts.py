
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from utils.data_loader import load_data, render_csv_uploader


st.set_page_config(
    page_title="Metrics and Alerts | RetailPulse",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Metrics, Alerts & Reports")
st.caption(
    "Historical sales monitoring, rule-based alerts, "
    "and downloadable reports."
)


# --------------------------------------------------
# 1. Load and refresh data
# --------------------------------------------------
render_csv_uploader()

def get_data():
    data = load_data().copy()

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )
    data["Revenue"] = pd.to_numeric(
        data["Revenue"], errors="coerce"
    )

    return data

if st.button("🔄 Refresh data"):
    get_data.clear()
    st.rerun()

try:
    df = get_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

required = {"InvoiceDate", "Revenue"}
if not required.issubset(df.columns):
    st.error("Dataset must contain InvoiceDate and Revenue columns.")
    st.stop()

df = df.dropna(subset=["InvoiceDate", "Revenue"]).copy()
if df.empty:
    st.warning("No valid dated revenue records were found.")
    st.stop()

df["Date"] = df["InvoiceDate"].dt.normalize()

daily = (
    df.groupby("Date", as_index=False)["Revenue"]
    .sum()
    .sort_values("Date")
)

latest_date = daily["Date"].max()
latest_revenue = float(
    daily.loc[daily["Date"] == latest_date, "Revenue"].iloc[0]
)

# Use the latest seven calendar dates present in the dataset.
recent = daily.tail(7).copy()
recent_avg = float(recent["Revenue"].mean())

# Compare the latest seven dates with the previous seven dates,
# when enough observations exist.
previous = daily.iloc[-14:-7]
previous_avg = (
    float(previous["Revenue"].mean())
    if len(previous) == 7
    else None
)

# --------------------------------------------------
# 2. Historical metrics
# --------------------------------------------------
st.subheader("Historical Sales Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Net Revenue",
    f"{df['Revenue'].sum():,.2f}"
)
c2.metric(
    "Latest Recorded Date",
    latest_date.strftime("%d %b %Y")
)
c3.metric(
    "Revenue on Latest Date",
    f"{latest_revenue:,.2f}"
)
c4.metric(
    "Average Revenue (Last 7 Records)",
    f"{recent_avg:,.2f}"
)

if previous_avg is not None:
    change = (
        (recent_avg - previous_avg) / abs(previous_avg) * 100
        if previous_avg != 0
        else None
    )

    if change is not None:
        st.metric(
            "Recent Average vs Previous 7 Records",
            f"{change:+.1f}%",
            help="Compares seven recorded dates with the preceding seven recorded dates."
        )

st.caption(
    f"Dataset coverage: {daily['Date'].min().date()} to "
    f"{latest_date.date()}. These are historical records, "
    "not a live transaction feed."
)

# --------------------------------------------------
# 3. Rule-based alerts
# --------------------------------------------------
st.subheader("⚠️ Sales Alerts")

alerts = []

threshold = st.slider(
    "Low-sales alert threshold (% below previous average)",
    min_value=10,
    max_value=80,
    value=30,
    step=5
)

if latest_revenue <= 0:
    alerts.append(
        ("Check latest date",
         f"Revenue on {latest_date.date()} is zero or negative.")
    )

if previous_avg is not None and previous_avg > 0:
    if recent_avg < previous_avg * (1 - threshold / 100):
        alerts.append(
            ("Low sales",
             f"Recent average revenue is more than {threshold}% "
             "below the previous seven recorded dates.")
        )

if not alerts:
    st.success("No alerts triggered by the current rules.")
else:
    for alert_title, message in alerts:
        st.warning(f"**{alert_title}:** {message}")

st.info(
    "Alerts are rule-based indicators for historical analysis. "
    "They do not confirm a live outage, current stock issue, "
    "or actual business incident."
)

# --------------------------------------------------
# 4. Historical revenue chart
# --------------------------------------------------
st.subheader("Daily Historical Revenue")

st.line_chart(
    daily.set_index("Date")["Revenue"]
)

with st.expander("View daily revenue records"):
    st.dataframe(daily, use_container_width=True, hide_index=True)

# --------------------------------------------------
# 5. CSV export
# --------------------------------------------------
st.subheader("Export Reports")

csv_bytes = daily.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Daily Revenue CSV",
    data=csv_bytes,
    file_name="retailpulse_daily_revenue.csv",
    mime="text/csv"
)

alerts_df = pd.DataFrame(
    alerts,
    columns=["Alert", "Details"]
)

if alerts_df.empty:
    alerts_df = pd.DataFrame(
        [{"Alert": "None", "Details": "No alert rules triggered."}]
    )

st.download_button(
    "Download Alerts CSV",
    data=alerts_df.to_csv(index=False).encode("utf-8"),
    file_name="retailpulse_alerts.csv",
    mime="text/csv"
)

# --------------------------------------------------
# 6. PDF report generation
# --------------------------------------------------
def create_pdf_report():
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(
        Paragraph("RetailPulse Sales Monitoring Report", styles["Title"])
    )
    story.append(Spacer(1, 12))

    generated_at = datetime.now().strftime("%d %b %Y, %H:%M")
    story.append(
        Paragraph(f"Report generated: {generated_at}", styles["Normal"])
    )
    story.append(
        Paragraph(
            f"Historical coverage: {daily['Date'].min().date()} "
            f"to {latest_date.date()}",
            styles["Normal"]
        )
    )
    story.append(Spacer(1, 12))

    summary_data = [
        ["Metric", "Value"],
        ["Total net revenue", f"{df['Revenue'].sum():,.2f}"],
        ["Latest recorded date", latest_date.strftime("%d %b %Y")],
        ["Revenue on latest date", f"{latest_revenue:,.2f}"],
        ["Average revenue (last 7 records)", f"{recent_avg:,.2f}"],
        ["Number of daily records", f"{len(daily):,}"],
        ["Alerts triggered", str(len(alerts))]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[260, 200],
        repeatRows=1
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#234E70")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 16))
    story.append(Paragraph("Alerts", styles["Heading2"]))

    for title, message in alerts:
        story.append(
            Paragraph(f"<b>{title}:</b> {message}", styles["BodyText"])
        )
        story.append(Spacer(1, 6))

    if not alerts:
        story.append(
            Paragraph("No alert rules were triggered.", styles["BodyText"])
        )

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Note: This report is based on historical transaction data. "
            "It is not a real-time operational monitoring report.",
            styles["Italic"]
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


if st.button("📄 Generate PDF Report"):
    try:
        pdf_data = create_pdf_report()
        st.download_button(
            "Download RetailPulse PDF Report",
            data=pdf_data,
            file_name="retailpulse_sales_report.pdf",
            mime="application/pdf"
        )
    except Exception as exc:
        st.error(f"PDF generation failed: {exc}")
