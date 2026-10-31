
from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st


# Project root: parent of streamlit_dashboard
PROJECT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "retail_cleaned.csv"


# Columns required for the dashboard's core retail analytics.
REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
    "IsCancelled",
    "Revenue",
]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize an uploaded or default retail dataset."""

    df = df.copy()

    # Remove accidental whitespace from column names.
    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():
        raise ValueError(
            "The CSV contains duplicate column names."
        )

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(missing)
        )

    if df.empty:
        raise ValueError("The CSV contains no data rows.")

    # Standardize types used by dashboard pages.
    raw_dates = df["InvoiceDate"]
    try:
        # `mixed` accepts files that contain more than one valid date
        # representation (for example, ISO timestamps and slash dates).
        parsed_dates = pd.to_datetime(
            raw_dates,
            format="mixed",
            errors="coerce",
        )
    except (TypeError, ValueError):
        # Compatibility with older pandas versions without format="mixed".
        parsed_dates = pd.to_datetime(raw_dates, errors="coerce")
    if parsed_dates.isna().all() and raw_dates.notna().any():
        # Some older pandas releases silently coerce an unsupported format.
        parsed_dates = pd.to_datetime(raw_dates, errors="coerce")
    df["InvoiceDate"] = parsed_dates

    for col in ["Quantity", "Price", "Revenue"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Reject files whose date or numerical fields cannot
    # be interpreted for any of the records.
    invalid_dates = df["InvoiceDate"].isna()
    if invalid_dates.any():
        examples = raw_dates[invalid_dates].astype(str).head(5).tolist()
        rows = (df.index[invalid_dates][:5] + 2).tolist()
        raise ValueError(
            f"{int(invalid_dates.sum())} InvoiceDate value(s) are invalid or "
            f"empty (CSV row(s): {rows}; examples: {examples}). "
            "Use a consistent date format such as YYYY-MM-DD HH:MM:SS."
        )

    if df[["Quantity", "Price", "Revenue"]].isna().any().any():
        raise ValueError(
            "Quantity, Price, and Revenue must contain valid "
            "numeric values in every row."
        )

    if df["StockCode"].isna().all():
        raise ValueError("StockCode cannot be entirely empty.")

    return df


def load_default_data() -> pd.DataFrame:
    """Load the original cleaned dataset from disk."""

    if not DEFAULT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Default dataset not found: {DEFAULT_DATA_PATH}"
        )

    df = pd.read_csv(DEFAULT_DATA_PATH)
    return prepare_data(df)


def load_data() -> pd.DataFrame:
    """
    Return the active dataset for the current Streamlit session.

    If a valid CSV was uploaded, use it.
    Otherwise, use the original cleaned dataset.
    """

    uploaded_df = st.session_state.get(
        "retailpulse_uploaded_data"
    )

    if uploaded_df is not None:
        return uploaded_df.copy()

    return load_default_data()


def render_csv_uploader() -> None:
    """Display the shared data source controls in the sidebar."""

    st.sidebar.divider()
    st.sidebar.subheader("📂 Data Source")

    uploaded_file = st.sidebar.file_uploader(
        "Upload retail CSV",
        type=["csv"],
        key="retailpulse_csv_uploader",
        help=(
            "Upload a replacement dataset. "
            "The original file on disk will not be modified."
        ),
    )
    st.sidebar.caption(
        "CSV needs Invoice, StockCode, Description, Quantity, InvoiceDate, "
        "Price, Customer ID, Country, IsCancelled, and Revenue columns."
    )

    if uploaded_file is not None:
        # Process only when a new file is selected.
        file_signature = (
            uploaded_file.name,
            uploaded_file.size,
            hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
        )

        previous_signature = st.session_state.get(
            "retailpulse_file_signature"
        )

        if file_signature != previous_signature:
            try:
                uploaded_file.seek(0)
                raw_df = pd.read_csv(uploaded_file)
                prepared_df = prepare_data(raw_df)

                # Commit only after validation succeeds.
                st.session_state[
                    "retailpulse_uploaded_data"
                ] = prepared_df

                st.session_state[
                    "retailpulse_file_signature"
                ] = file_signature

                st.session_state[
                    "retailpulse_data_source_name"
                ] = uploaded_file.name

                st.session_state[
                    "retailpulse_upload_error"
                ] = None

                st.rerun()

            except Exception as exc:
                st.session_state[
                    "retailpulse_upload_error"
                ] = str(exc)

    error = st.session_state.get(
        "retailpulse_upload_error"
    )

    if error:
        st.sidebar.error(f"Upload failed: {error}")

    if st.session_state.get("retailpulse_uploaded_data") is not None:
        source_name = st.session_state.get(
            "retailpulse_data_source_name",
            "Uploaded CSV"
        )

        st.sidebar.success(f"Active dataset: {source_name}")

        st.sidebar.button(
            "Use original dataset",
            key="retailpulse_reset_data",
            on_click=reset_uploaded_data,
        )
    else:
        st.sidebar.info("Using original cleaned dataset")


def reset_uploaded_data() -> None:
    """Clear the uploaded data and uploader so the default is restored."""

    for key in (
        "retailpulse_uploaded_data",
        "retailpulse_file_signature",
        "retailpulse_data_source_name",
        "retailpulse_upload_error",
        "retailpulse_csv_uploader",
    ):
        st.session_state.pop(key, None)
