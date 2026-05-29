# ============================================================
# src/forecast.py — TBATS Forecasting Engine
# ============================================================
# FIXED: Model is now loaded inside generate_forecast() on
# every call, not once at module import. This ensures that
# after retraining saves a new .pkl, the next forecast call
# automatically picks up the updated model without needing
# a Streamlit server restart.
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TBATS_MODEL_PATH
from src.database import get_all_sales


def generate_forecast(horizon_days=7):
    """
    Generates a sales forecast for the next N days using TBATS.

    Model is loaded fresh on every call so that a newly saved
    model after retraining is picked up immediately.

    Parameters:
        horizon_days (int): Number of days to forecast (7, 14, 30).

    Returns:
        DataFrame with columns:
            - 'date'           : forecast dates (Timestamp)
            - 'predicted_sales': predicted total daily sales (float)
    """

    # ── Load Model Fresh (picks up retrained model immediately) ──
    # Loading inside the function means any new .pkl saved by
    # the retrain page is used on the very next forecast call.
    tbats_model = joblib.load(TBATS_MODEL_PATH)

    # ── Step 1: Load All Sales Data from Database ─────────────
    df = get_all_sales()
    df = df.sort_values('date').reset_index(drop=True)

    # ── Step 2: Build a Time-Indexed Series ───────────────────
    sales_series = df.set_index('date')['sales']

    # ── Step 3: Generate Forecast ─────────────────────────────
    # tbats_model.forecast(steps=N) produces N future values
    # starting from the day after the last date in the series.
    forecast_values = tbats_model.forecast(steps=horizon_days)

    # Clip negatives — sales cannot be below 0
    forecast_values = np.maximum(forecast_values, 0)

    # ── Step 4: Build Future Date Range ───────────────────────
    last_date = df['date'].max()

    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(days=1),
        periods=horizon_days,
        freq='D'
    )

    # ── Step 5: Return Forecast DataFrame ─────────────────────
    forecast_df = pd.DataFrame({
        'date':            forecast_dates,
        'predicted_sales': [round(float(v), 2) for v in forecast_values]
    })

    return forecast_df


def get_recent_history(n_days=90):
    """
    Returns the last n_days of daily sales from the database.
    Used by the Forecast page to plot historical context
    before the forecast line begins on the chart.

    Parameters:
        n_days (int): Number of recent days to return. Default=90

    Returns:
        DataFrame with columns: ['date', 'sales']
    """
    df = get_all_sales()
    df = df.sort_values('date').reset_index(drop=True)
    return df.tail(n_days)
