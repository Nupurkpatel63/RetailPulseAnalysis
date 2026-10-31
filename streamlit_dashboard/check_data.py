
from utils.data_loader import load_retail_data

df = load_retail_data()

print("\n--- DATASET OVERVIEW ---")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum().sort_values(ascending=False))

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- DATE VALIDATION ---")
if "InvoiceDate" in df.columns:
    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    print("Invalid dates:", dates.isna().sum())
    print("Earliest date:", dates.min())
    print("Latest date:", dates.max())

print("\n--- BUSINESS METRICS ---")
if "Quantity" in df.columns:
    quantity = pd.to_numeric(df["Quantity"], errors="coerce")
    print("Total quantity:", quantity.sum())
    print("Positive units:", quantity.clip(lower=0).sum())
    print("Negative quantity rows:", (quantity < 0).sum())

if "Revenue" in df.columns:
    revenue = pd.to_numeric(df["Revenue"], errors="coerce")
elif {"Quantity", "Price"}.issubset(df.columns):
    revenue = (
        pd.to_numeric(df["Quantity"], errors="coerce")
        * pd.to_numeric(df["Price"], errors="coerce")
    )
else:
    revenue = None

if revenue is not None:
    print("Total revenue:", revenue.sum())
    print("Negative revenue rows:", (revenue < 0).sum())

if "Invoice" in df.columns:
    print("Unique invoices:", df["Invoice"].nunique())

if "Customer ID" in df.columns:
    print("Unique customers:", df["Customer ID"].nunique())
