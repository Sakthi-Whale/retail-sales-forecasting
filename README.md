# 🛒 Retail Sales Forecasting Dashboard

> AI-Powered Sales Intelligence using the **TBATS Time Series Model**  
> Corporación Favorita Ecuador Dataset · 2019 – 2023

---

## 📌 Project Overview

This is a capstone project for **Retail Sales Forecasting** built on the Corporación Favorita Ecuador dataset. It covers the complete end-to-end machine learning pipeline — from raw data preprocessing to a fully interactive forecasting dashboard with a live model retraining loop.

The system forecasts **daily total sales** using the TBATS model, which achieved the best performance across all 5 models evaluated (R² = 0.9951, RMSE = 1,405.30).

---

## 🏆 Model Comparison

| Rank | Model | RMSE | R² Score | Status |
|------|-------|------|----------|--------|
| 🥇 1 | **TBATS** | 1,405.30 | 0.9951 | ✅ Selected |
| 🥈 2 | Random Forest | 1,511.23 | 0.9944 | ❌ Rejected |
| 🥉 3 | ARIMA | 17,605.24 | 0.2310 | ❌ Rejected |
| 4 | Prophet | 22,811.88 | -0.2912 | ❌ Rejected |
| 5 | LSTM | 6,741,163,939.49 | -16.0047 | ❌ Rejected |

---

## 🗂️ Project Structure

```
retail_forecasting/
│
├── app.py                    # Main Streamlit entry point (Home page)
├── config.py                 # Central path & settings configuration
├── requirements.txt          # Python dependencies
├── forecasting.db            # SQLite database (daily sales records)
│
├── src/
│   ├── __init__.py           # Package initializer
│   ├── database.py           # All database CRUD operations
│   └── forecast.py           # TBATS forecasting engine
│
├── pages/
│   ├── 1_Dashboard.py        # Historical sales charts & trends
│   ├── 2_Forecast.py         # 7 / 14 / 30-day forecast generator
│   ├── 3_Daily_Update.py     # Manual daily data entry page
│   ├── 4_Retrain.py          # TBATS model retraining pipeline
│   └── 5_Delete_Record.py    # Delete a record by date
│
├── models/
│   └── tbats_model.pkl       # Saved TBATS model (joblib)
│
└── processed/
    └── tbats_train_data.csv  # Aggregated daily sales (date + sales)
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/retail-sales-forecasting.git
cd retail-sales-forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

> **Note:** On first launch, the database is automatically seeded from `processed/tbats_train_data.csv`. No manual setup is required.

---

## 📦 Dependencies

```
streamlit==1.32.0
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.4.0
joblib==1.3.2
plotly==5.18.0
sqlalchemy==2.0.25
tbats
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📄 Pages & Features

### 🏠 Home (`app.py`)
- Project overview and model comparison table
- TBATS explainer (what each letter stands for)
- Navigation guide to all pages

### 📊 Dashboard (`pages/1_Dashboard.py`)
- Historical sales trends from 2019–2023
- Year-on-year comparisons and monthly breakdowns
- Powered by Plotly interactive charts

### 🔮 Forecast (`pages/2_Forecast.py`)
- Generate **7, 14, or 30-day** sales forecasts
- Combined chart showing last 90 days of history + forecast line
- Day-by-day forecast table with downloadable CSV export

### 📅 Daily Update (`pages/3_Daily_Update.py`)
- Manually add a new date and actual sales figure to the database
- Simulates a real-world POS (Point-of-Sale) data pipeline
- Existing dates are **overwritten** (upsert behaviour)

### 🔁 Retrain (`pages/4_Retrain.py`)
- Fair holdout evaluation: both old and new TBATS models are tested on the **same last 30 days**
- Visual comparison chart (Actual vs Old vs New predictions)
- If new model wins → automatically retrained on **100% of data** and saved

### 🗑️ Delete Record (`pages/5_Delete_Record.py`)
- Delete any daily sales record by selecting its date
- Shows a preview of the record before deletion
- Requires confirmation checkbox to prevent accidental deletes

---

## 🤖 Why TBATS?

**TBATS** stands for:

| Letter | Component | Purpose |
|--------|-----------|---------|
| **T** | Trigonometric Seasonality | Fourier terms for complex multi-frequency seasonality |
| **B** | Box-Cox Transformation | Stabilises variance in sales data |
| **A** | ARMA Errors | Captures residual autocorrelation |
| **T** | Trend Component | Models long-term upward/downward drift |
| **S** | Seasonal Components | Weekly (7-day) + yearly (365-day) cycles |

Retail sales data has strong **weekly cycles** (weekend spikes) and **yearly cycles** (holidays, festive seasons) — exactly the patterns TBATS is designed to handle simultaneously, without any manual feature engineering.

---

## 🗄️ Database Schema

The app uses a lightweight **SQLite** database (`forecasting.db`) with a single table:

```sql
CREATE TABLE daily_sales (
    date  TEXT PRIMARY KEY,   -- Format: YYYY-MM-DD
    sales REAL                -- Total daily sales across all stores
);
```

All database operations are handled in `src/database.py`:

| Function | Operation | Description |
|----------|-----------|-------------|
| `init_database()` | Create | Creates table and seeds from CSV on first run |
| `get_all_sales()` | Read | Returns all records as a sorted DataFrame |
| `get_latest_date()` | Read | Returns the most recent date in the database |
| `append_new_day()` | Create/Update | Inserts or overwrites a daily record |
| `date_exists()` | Read | Checks if a date already exists |

---

## 🔁 Retraining Pipeline

The retraining logic uses a **fair holdout evaluation** approach:

```
All Data
├── Eval-Train Split  (everything except last 30 days)
│   └── Train NEW model → evaluate on holdout
├── Holdout (last 30 days) ← same window for both models
│   └── OLD model forecasts these 30 days
│   └── NEW model forecasts these 30 days
│
If NEW model RMSE < OLD model RMSE:
    └── Retrain NEW model on 100% of data → save to tbats_model.pkl
Else:
    └── Keep OLD model unchanged
```

This ensures the comparison is **always fair** and the saved model always learns from the **maximum available data**.

---

## 📊 Dataset

- **Source:** Corporación Favorita Ecuador (Kaggle)
- **Scope:** 1,700+ stores across multiple product families
- **Aggregation:** Daily total sales across all stores
- **Training Range:** 2019 – 2023
- **Format:** `date` (YYYY-MM-DD), `sales` (float)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Forecasting | TBATS (`tbats` library) |
| Data | pandas, numpy |
| Database | SQLite3 |
| Charts | Plotly |
| ML Metrics | scikit-learn |
| Model Persistence | joblib |

---

## 👨‍💻 Author

Capstone Project — Retail Sales Forecasting  
Built with ❤️ using Python, Streamlit & TBATS
