# RetailPulse — Retail Sales Analytics & Forecasting

RetailPulse is a retail analytics project built around transaction data. It includes a multi-page Streamlit dashboard, notebooks for data exploration and machine-learning workflows, and saved model/report artifacts.

## Features

The dashboard in `streamlit_dashboard/` provides:

- **Overview:** net revenue, orders, customers, products, and daily revenue trends.
- **Demand forecasting:** forecasts from a saved LSTM model, forecast horizons of 7, 14, or 30 days, percentage what-if scenarios, CSV export, and optional retraining.
- **Customer analytics:** RFM-style customer metrics and rule-based customer segments.
- **Sales performance:** date filtering, daily and monthly revenue, top products, country comparisons, and CSV export.
- **Model performance:** forecast evaluation and error metrics using available evaluation data.
- **Inventory optimization:** demand-based reorder estimates using configurable lead time, stock coverage, and safety buffer.
- **Metrics and alerts:** historical revenue monitoring, rule-based sales alerts, and CSV/PDF reports.
- **CSV uploads:** use a replacement dataset for the current dashboard session; the original data file is not overwritten.

The notebooks cover time-series exploration, Prophet and LSTM forecasting, churn prediction with XGBoost and SHAP, feature importance and Optuna tuning, inventory optimization, drift detection with Evidently, and an Airflow retraining workflow.

## Project layout

```text
.
├── data/                  # Retail transaction data and derived CSVs
├── notebook/              # Analysis, modeling, drift, and retraining notebooks
├── streamlit_dashboard/   # Streamlit app, pages, and dashboard utilities
├── model/                 # Saved LSTM model and revenue scaler
├── churn_model/           # Saved churn model, feature columns, and evaluation outputs
├── airflow_retraining/    # Champion/candidate model artifacts and run metadata
├── optuna_outputs/        # Tuning results and feature-importance outputs
├── drift_outputs/         # Generated drift report
└── output/                # Project portfolio PDF
```

## Requirements

- Python 3.10 or newer is recommended.
- Install dashboard and analysis dependencies listed in `streamlit_dashboard/requirements.txt`.
- The forecasting page needs the saved LSTM model and scaler in `model/`.

## Run the dashboard

From the project root:

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies and launch Streamlit:

```bash
python -m pip install -r streamlit_dashboard/requirements.txt
streamlit run streamlit_dashboard/app.py
```

Streamlit will print a local URL to open in your browser.

## Data format

The default dashboard dataset is `data/retail_cleaned.csv`. Dashboard CSV uploads must include these columns:

| Column | Purpose |
| --- | --- |
| `Invoice` | Invoice/order identifier |
| `StockCode` | Product identifier |
| `Description` | Product description |
| `Quantity` | Units on the transaction line |
| `InvoiceDate` | Transaction date and time |
| `Price` | Unit price |
| `Customer ID` | Customer identifier |
| `Country` | Customer country |
| `IsCancelled` | Cancellation indicator |
| `Revenue` | Net revenue for the transaction line |

Dates must be parseable and `Quantity`, `Price`, and `Revenue` must be numeric. The uploader validates the data before using it. Uploaded data is held in Streamlit session state and is not saved over the default CSV.

## Forecasting notes

The LSTM forecast uses daily aggregated revenue, fills calendar days with no transactions with zero revenue, and predicts recursively from the most recent 28 days. The bundled model and scaler are loaded from `model/lstm_revenue_model.pth` and `model/revenue_scaler.pkl`. Retraining requires at least 38 daily observations and saves updated artifacts to those paths.

A forecast is a model estimate based on the available history. Inventory recommendations are simple estimates based on observed positive sales; current stock must be entered by the user. Alerts are historical, rule-based indicators rather than live operational monitoring.

## Notebooks

Open the notebooks under `notebook/` in Jupyter or VS Code to explore and rerun the analysis. Some notebooks may rely on their own data paths, generated artifacts, or external services such as Airflow; review each notebook's setup cells before running it.

## License and data

No license file is currently included. Confirm that you have permission to use and redistribute the transaction data and generated artifacts before sharing this repository.
