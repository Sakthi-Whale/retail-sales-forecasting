# ============================================================
# pages/2_Forecast.py — Sales Forecast Page
# ============================================================
# This page allows the user to generate future sales forecasts
# using the trained TBATS (Trigonometric Boxcox ARMA Trend
# Seasonal) model.
#
# User selects:
#   - Forecast horizon: 7, 14, or 30 days
#
# The app then:
#   1. Loads all current data from the SQLite database
#   2. Passes the full sales history to the TBATS model
#   3. TBATS auto-generates N future predictions (multi-step)
#   4. Displays a combined chart (past 90 days + forecast line)
#   5. Shows a summary table with day-by-day forecasted values
#
# ✅ Updated for TBATS:
#    - No lag/rolling feature engineering needed
#    - TBATS handles trend + seasonality natively
#    - generate_forecast() calls tbats_model.forecast(steps=N)
# ============================================================

import streamlit as st
# streamlit → web UI library. Used for all UI elements.

import pandas as pd
# pandas → data manipulation library. Used to handle
# DataFrames for historical and forecasted data.

import plotly.graph_objects as go
# plotly.graph_objects → used for creating advanced charts
# with multiple traces (e.g., actual + forecast on same chart)

import os
import sys

# Add root folder to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TBATS forecast generation function
# generate_forecast() returns a DataFrame with date + predicted_sales
from src.forecast import generate_forecast, get_recent_history

# Import database function to get the latest date for display
from src.database import get_latest_date

# Import app settings
from config import APP_TITLE

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title=f"Forecast | {APP_TITLE}",
    page_icon="🔮",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .page-header {
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            padding: 2rem; border-radius: 14px;
            margin-bottom: 1.5rem; text-align: center;
        }
        .page-header h2 { color: #e2e8f0; font-size: 1.9rem; margin: 0; }
        .page-header p  { color: #94a3b8; margin-top: 0.3rem; }
        .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ── Page Header ──────────────────────────────────────────────
st.markdown("""
    <div class="page-header">
        <h2>🔮 Sales Forecast</h2>
        <p>Generate future sales predictions using the TBATS time series model</p>
    </div>
""", unsafe_allow_html=True)


# ── About TBATS Forecasting ──────────────────────────────────
# Collapsed explanation box — users can expand if curious
with st.expander("ℹ️ How does TBATS forecasting work?", expanded=False):
    st.markdown("""
    **TBATS (Trigonometric, Box-Cox, ARMA, Trend, Seasonal)** is a
    state-space time series model designed to handle complex seasonality.

    **Forecasting steps:**
    1. The model reads **all sales records** from the database as its history
    2. It calls `tbats_model.forecast(steps=N)` to predict N future days
    3. TBATS internally uses its learned trend + seasonal components
    4. No manual feature engineering or lag creation is required

    **Key advantage:** TBATS can model multiple seasonal patterns (weekly,
    yearly) simultaneously — making it ideal for retail sales data with
    strong weekly weekday/weekend cycles and annual holiday peaks.
    """)

st.divider()

# ── Forecast Controls ────────────────────────────────────────
st.subheader("⚙️ Forecast Settings")

# Show the latest date from which forecasting will begin.
# get_latest_date() queries MAX(date) from the SQLite database.
latest_date = get_latest_date()

# Convert to datetime if it's a string so we can format it
latest_date = pd.to_datetime(latest_date)

# st.info() shows a blue information box with current status
st.info(
    f"📅 **Forecasting from:** {str(latest_date)[:10]}  \n"
    f"The model will predict daily sales starting from the day after this date."
)

# st.radio() creates a row of selectable button options.
# The user picks one of the three forecast horizons.
# horizontal=True displays the buttons side by side.
horizon = st.radio(
    "Select Forecast Horizon:",
    options=[7, 14, 30],
    format_func=lambda x: f"{x} Days",  # display label for each option
    horizontal=True
)

st.divider()

# ── Generate Forecast Button ─────────────────────────────────
# st.button() creates a clickable button.
# Returns True only when the button is clicked by the user.
if st.button("🚀 Generate Forecast", use_container_width=True, type="primary"):

    # st.spinner() shows a loading animation while the
    # code inside the 'with' block is running.
    with st.spinner(f"Running TBATS model — generating {horizon}-day forecast..."):

        try:
            # ── Generate Forecast ─────────────────────────────
            # generate_forecast() loads all DB records, calls
            # tbats_model.forecast(steps=horizon), and returns
            # a DataFrame with columns: date, predicted_sales
            forecast_df = generate_forecast(horizon_days=horizon)

            # ── Get Historical Context for Chart ─────────────
            # get_recent_history(90) returns the last 90 days
            # of actual sales from the database for the chart's
            # historical section (the blue line before forecast)
            history_df = get_recent_history(n_days=90)

            # ── Build Combined Chart ─────────────────────────
            # One chart with two lines:
            # Line 1 (blue)        → last 90 days of actual sales
            # Line 2 (orange, dot) → forecasted future sales

            # go.Figure() creates an empty chart to add traces to
            fig = go.Figure()

            # Trace 1: Historical actual sales (solid blue line)
            # x = dates from the last 90 days
            # y = actual sales values from the database
            fig.add_trace(go.Scatter(
                x=history_df['date'],
                y=history_df['sales'],
                mode='lines',
                name='Historical Sales',
                line=dict(color='#3b82f6', width=2),  # solid blue
            ))

            # Trace 2: TBATS forecast line (orange dotted with markers)
            # mode='lines+markers' draws the line AND a dot at each
            # predicted point — useful to distinguish individual days
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['predicted_sales'],
                mode='lines+markers',
                name=f'{horizon}-Day TBATS Forecast',
                line=dict(color='#f59e0b', width=2.5, dash='dot'),  # orange dotted
                marker=dict(size=6)  # dot size at each prediction point
            ))

            # add_vline() draws a vertical dashed grey line at the
            # exact point where history ends and forecast begins.
            # This makes the chart easy to read at a glance.
            #
            # We convert the last history date to Unix milliseconds
            # (a plain float) because plotly's add_vline() in newer
            # pandas versions requires a numeric value, not Timestamp.
            vline_ts = pd.to_datetime(history_df['date'].iloc[-1]).timestamp() * 1000

            fig.add_vline(
                x=vline_ts,                        # position in ms
                line_dash="dash",                  # dashed style
                line_color="#64748b",              # muted grey color
                annotation_text="Forecast Start",  # label text
                annotation_position="top"          # place label at top
            )

            # Apply dark professional styling to the chart
            fig.update_layout(
                title=f'TBATS Sales Forecast — Next {horizon} Days',
                xaxis_title='Date',
                yaxis_title='Total Sales',
                plot_bgcolor='#0f172a',
                paper_bgcolor='#0f172a',
                font_color='#94a3b8',
                legend=dict(
                    bgcolor='#1e293b',
                    bordercolor='#334155',
                    borderwidth=1
                ),
                hovermode='x unified',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#1e293b')
            )

            # Render the plotly chart inline in Streamlit
            st.plotly_chart(fig, use_container_width=True)

            # ── Forecast Summary Metrics ─────────────────────
            # Display 3 key numbers below the chart for quick
            # reading without scrolling through the full table.
            st.subheader("📊 Forecast Summary")

            s1, s2, s3 = st.columns(3)

            with s1:
                # Show the date range the forecast covers
                # iloc[0] → first forecast date
                # iloc[-1] → last forecast date
                st.metric(
                    "📅 Forecast Period",
                    f"{str(forecast_df['date'].iloc[0])[:10]}  →  "
                    f"{str(forecast_df['date'].iloc[-1])[:10]}"
                )
            with s2:
                # Total of all predicted daily sales across horizon
                st.metric(
                    "💰 Total Forecasted Sales",
                    f"{forecast_df['predicted_sales'].sum():,.0f}"
                )
            with s3:
                # Average predicted daily sales over the horizon
                st.metric(
                    "📈 Avg Daily Forecast",
                    f"{forecast_df['predicted_sales'].mean():,.0f}"
                )

            # ── Forecast Table ───────────────────────────────
            # Show the full day-by-day breakdown as an interactive
            # scrollable table so users can see exact values.
            st.subheader("📋 Day-by-Day Forecast Table")

            # Format a copy of the forecast DataFrame for display.
            # We copy() to avoid modifying the original data.
            display_forecast = forecast_df.copy()

            # Convert date to plain YYYY-MM-DD string
            display_forecast['date'] = display_forecast['date'].astype(str).str[:10]

            # Format sales as comma-separated with 2 decimal places
            # e.g. 73521.0 → "73,521.00"
            display_forecast['predicted_sales'] = display_forecast['predicted_sales'].apply(
                lambda x: f"{x:,.2f}"
            )

            # Rename columns for user-friendly display
            display_forecast.columns = ['Date', 'Predicted Sales']

            # st.dataframe() renders an interactive scrollable table
            st.dataframe(display_forecast, hide_index=True, use_container_width=True)

            # ── Download Button ──────────────────────────────
            # Let users download the forecast data as a CSV file.
            # to_csv(index=False) converts the DataFrame to CSV
            # without the pandas row number column.
            csv_data = forecast_df.to_csv(index=False)

            st.download_button(
                label="⬇️ Download Forecast as CSV",
                data=csv_data,
                file_name=f"tbats_forecast_{horizon}days.csv",
                mime="text/csv"
            )

        except Exception as e:
            # If anything fails, show the full error and traceback
            # so the user knows exactly what went wrong.
            import traceback
            st.error(f"❌ Forecast failed: {e}")
            # st.code() shows the full Python traceback with file
            # names and line numbers — essential for debugging.
            st.code(traceback.format_exc())
