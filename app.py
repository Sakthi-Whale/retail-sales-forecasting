# ============================================================
# app.py — Main Streamlit Application Entry Point
# ============================================================
# This is the first file Streamlit runs when you execute:
#     streamlit run app.py
#
# It is responsible for:
#   1. Configuring the overall page layout and theme
#   2. Initializing the SQLite database on first launch
#      (loads tbats_train.csv into daily_sales table)
#   3. Displaying the home/welcome page of the dashboard
#   4. Providing navigation instructions to other pages
#
# ✅ Updated for TBATS model:
#    - No store-level features or simulation CSV needed
#    - Database is seeded from tbats_train.csv (date + sales)
#    - Model comparison table updated with correct TBATS metrics
# ============================================================

import streamlit as st
# streamlit → the main library used to build the web app UI.
# It converts Python scripts into interactive web pages
# without needing HTML, CSS, or JavaScript knowledge.
# Every st.xxx() call creates a UI element on the page.

import pandas as pd
# pandas → data manipulation library.
# Used here to load and display data summaries on the home page.

import os
# os → built-in module for file and path operations.
# Used to check if files and folders exist before loading them.

import sys
# sys → built-in module for system-level operations.
# Used to add the current directory to Python's search path
# so all other modules (config, src, pages) can be found.

# Ensure the root project folder is in Python's module search path.
# This allows imports like 'from config import ...' to work
# from any file inside the project.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the database initialization function from our
# custom database module in the src folder
from src.database import init_database

# Import path constants and app settings from config.py
from config import APP_TITLE, APP_ICON, DB_PATH

# ── Page Configuration ───────────────────────────────────────
# st.set_page_config() MUST be the very first Streamlit command.
# It sets global settings for the entire app.
st.set_page_config(
    page_title=APP_TITLE,       # Title shown in browser tab
    page_icon=APP_ICON,         # Icon shown in browser tab
    layout="wide",              # 'wide' uses full screen width
    initial_sidebar_state="expanded"  # sidebar open by default
)

# ── Custom CSS Styling ───────────────────────────────────────
# st.markdown() with unsafe_allow_html=True lets us inject
# raw HTML and CSS into the Streamlit page for custom styling.
# This gives us control over colors, fonts, and spacing
# that Streamlit's built-in themes don't provide.
st.markdown("""
    <style>
        /* Import Google Font — DM Sans for professional look */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

        /* Apply font to entire app */
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }

        /* Style the main header banner */
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }

        .main-header h1 {
            color: #e2e8f0;
            font-size: 2.4rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.5px;
        }

        .main-header p {
            color: #94a3b8;
            font-size: 1.05rem;
            margin-top: 0.5rem;
        }

        /* Style metric cards */
        .metric-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }

        /* Style info boxes used in navigation cards */
        .info-box {
            background: #0f172a;
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin: 0.5rem 0;
            color: #cbd5e1;
        }

        /* Remove default Streamlit top padding */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


# ── Initialize Database ──────────────────────────────────────
# This runs the database setup every time the app starts.
# init_database() checks if the daily_sales table is empty —
# if yes, it loads tbats_train.csv and inserts all records.
# If the table already has data, it skips the loading step
# so existing data (including user-added days) is preserved.
try:
    init_database()
except Exception as e:
    # st.error() displays a red error box in the UI
    st.error(f"⚠️ Database initialization failed: {e}")
    st.stop()  # Stop the app if database can't be initialized


# ── Home Page Header ─────────────────────────────────────────
# Display the main title banner using our custom CSS class
st.markdown("""
    <div class="main-header">
        <h1>🛒 Retail Sales Forecasting Dashboard</h1>
        <p>AI-Powered Sales Intelligence — TBATS Time Series Model</p>
    </div>
""", unsafe_allow_html=True)


# ── Project Overview Section ─────────────────────────────────
# st.columns() splits the page into side-by-side columns.
# (1, 1) means two equal-width columns.
col1, col2 = st.columns(2)

with col1:
    # st.subheader() creates a smaller section heading
    st.subheader("📌 About This Project")

    # st.markdown() renders formatted text with markdown syntax
    st.markdown("""
    This dashboard is the capstone project for **Retail Sales Forecasting**
    using machine learning on the Corporación Favorita Ecuador dataset.

    The project covers the complete ML pipeline:
    - 📦 Data collection and preprocessing
    - 📊 Time series analysis and stationarity testing
    - 🤖 Model training and evaluation (ARIMA, Prophet, RF, LSTM, TBATS)
    - 📈 Interactive forecasting dashboard
    - 🔄 Live daily data entry and model retraining
    """)

with col2:
    st.subheader("🏆 Model Performance")

    st.markdown("""
    Five models were trained and evaluated on this dataset:
    """)

    # st.dataframe() renders an interactive table in the UI.
    # pd.DataFrame() creates a table from a dictionary.
    # The TBATS model achieved the best RMSE and R² overall.
    model_results = pd.DataFrame({
    'Rank': [
        '🥇 1',
        '🥈 2',
        '🥉 3',
        '4',
        '5'
    ],

    'Model': [
        '✅ TBATS',
        'Random Forest',
        'ARIMA',
        'Prophet',
        'LSTM'
    ],

    'RMSE': [
        1405.30,
        1511.23,
        17605.24,
        22811.88,
        6741163939.49
    ],

    'R² Score': [
        0.9951,
        0.9944,
        0.2310,
        -0.2912,
        -16.0047
    ],

    'Status': [
        '🏆 Selected',
        '❌ Rejected',
        '❌ Rejected',
        '❌ Rejected',
        '❌ Rejected'
    ]
})
    # hide_index=True removes the row number column from display
    st.dataframe(model_results, hide_index=True, use_container_width=True)


# ── Divider ──────────────────────────────────────────────────
# st.divider() draws a horizontal line to separate sections
st.divider()


# ── Navigation Guide ─────────────────────────────────────────
st.subheader("🧭 Navigate the Dashboard")

# Create 4 equal columns for the 4 navigation cards
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="info-box">
        <h4>📊 Dashboard</h4>
        <p>View historical sales trends, year-over-year comparisons,
        and monthly breakdowns from the training data</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-box">
        <h4>🔮 Forecast</h4>
        <p>Generate 7, 14, or 30-day sales forecasts using the
        trained TBATS model on all available database records</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-box">
        <h4>📅 Daily Update</h4>
        <p>Manually enter a new date and actual sales figure to
        extend the database with fresh real-world observations</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="info-box">
        <h4>🔁 Retrain</h4>
        <p>Retrain the TBATS model on the updated database and
        compare old vs new model performance before saving</p>
    </div>
    """, unsafe_allow_html=True)


# ── Dataset Info ─────────────────────────────────────────────
st.divider()
st.subheader("📁 Dataset Information")

# Create 3 columns for dataset statistics
d1, d2, d3 = st.columns(3)

with d1:
    # st.metric() creates a styled metric display card
    # label = title, value = main number shown prominently
    st.metric(label="📅 Training Data", value="2019 – 2023")

with d2:
    st.metric(label="🤖 Production Model", value="TBATS")

with d3:
    st.metric(label="🏆 Model R² Score", value="0.9951")


# ── Why TBATS Section ─────────────────────────────────────────
st.divider()
st.subheader("🔍 Why TBATS?")

# Use a 2-column layout for the TBATS explanation
t1, t2 = st.columns(2)

with t1:
    st.markdown("""
    **TBATS** stands for:
    - **T** — Trigonometric seasonality (Fourier terms)
    - **B** — Box-Cox transformation (handles variance)
    - **A** — ARMA errors (captures autocorrelation)
    - **T** — Trend component (upward/downward drift)
    - **S** — Seasonal components (weekly, yearly cycles)

    TBATS is purpose-built for complex time series with
    multiple seasonalities and non-stationary behavior —
    exactly the patterns present in retail sales data.
    """)

with t2:
    st.markdown("""
    **Why TBATS outperformed other models:**

    - ✅ **Highest R²** of 0.9951 across all 5 models
    - ✅ **Lowest RMSE** of 1,405.30 (smallest prediction error)
    - ✅ Handles multiple seasonal cycles simultaneously
    - ✅ No manual feature engineering required
    - ✅ Automatically detects trend and seasonality
    - ✅ Produces multi-step forecasts natively
    """)


# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown("""
    <p style='text-align:center; color:#475569; font-size:0.85rem;'>
        Retail Sales Forecasting — Capstone Project &nbsp;|&nbsp;
        Built with Streamlit & TBATS &nbsp;|&nbsp;
        Corporación Favorita Ecuador Dataset
    </p>
""", unsafe_allow_html=True)
