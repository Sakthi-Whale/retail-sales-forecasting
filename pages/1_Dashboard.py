# ============================================================
# pages/1_Dashboard.py — Sales Overview Dashboard Page
# ============================================================
# Streamlit automatically detects files inside the 'pages/'
# folder and adds them as separate pages in the sidebar.
# The number prefix (1_) controls the order in the sidebar.
#
# This page shows historical sales insights from the database
# (which is seeded from tbats_train.csv). Charts and metrics
# are dynamically built from all current database records,
# including any new days added via the Daily Update page.
#
# This page shows:
#   1. Key summary metrics (total sales, date range, avg daily)
#   2. Full historical sales trend line chart
#   3. Monthly sales bar chart grouped by year
#   4. Year-over-year sales comparison (bar + summary table)
#
# ✅ Updated for TBATS: data comes directly from SQLite database
#    which is loaded from tbats_train.csv (date + sales only).
# ============================================================

import streamlit as st
# streamlit → the main library for building the web UI.
# All UI elements (charts, text, tables) are created with st.xxx()

import pandas as pd
# pandas → data manipulation library.
# Used to load, filter, group, and aggregate sales data
# for display in charts and tables.

import plotly.express as px
# plotly.express → high-level charting library for creating
# interactive charts (line charts, bar charts, etc.)
# Charts made with plotly are interactive — users can zoom,
# hover for details, and download the chart as an image.

import plotly.graph_objects as go
# plotly.graph_objects → lower-level plotly API for more
# customized and complex chart configurations (used here
# for the sales decomposition chart with multiple traces).

# ── Model Performance Rankings (from notebook evaluation) ────
# Updated with actual results after fixing the TBATS data pipeline.
# LSTM ranked last due to scaler leakage inflating its error massively.
MODEL_RANKINGS = [
    {"rank": 1, "model": "TBATS",         "mse": 1_974_873.95,        "rmse": 1405.30,         "r2":  0.9951,   "medal": "🥇"},
    {"rank": 2, "model": "Random Forest", "mse": 2_283_820.94,        "rmse": 1511.23,         "r2":  0.9944,   "medal": "🥈"},
    {"rank": 3, "model": "ARIMA",         "mse": 309_944_612.66,      "rmse": 17605.24,        "r2":  0.2310,   "medal": "🥉"},
    {"rank": 4, "model": "Prophet",       "mse": 520_381_656.71,      "rmse": 22811.88,        "r2": -0.2912,   "medal": "🎖️"},
    {"rank": 5, "model": "LSTM",          "mse": 4.54e19,             "rmse": 6_741_163_939.49,"r2": -16.0047,  "medal": "🎖️"},
]

import os
import sys

# Add root project folder to Python's search path so we can
# import modules in the 'src' and root folders correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database function to get all sales data from SQLite
from src.database import get_all_sales, get_latest_date

# Import app settings from config
from config import APP_TITLE, APP_ICON

# ── Page Configuration ───────────────────────────────────────
# Must be called before any other Streamlit command on this page
st.set_page_config(
    page_title=f"Dashboard | {APP_TITLE}",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
# Injects Google Font and custom styles for this page's header
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .page-header {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            padding: 2rem;
            border-radius: 14px;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .page-header h2 { color: #e2e8f0; font-size: 1.9rem; margin: 0; }
        .page-header p  { color: #94a3b8; margin-top: 0.3rem; }
        .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ── Page Header ──────────────────────────────────────────────
# Renders the dark teal gradient header banner at the top
st.markdown("""
    <div class="page-header">
        <h2>📊 Sales Overview Dashboard</h2>
        <p>Historical sales trends and performance insights from the TBATS training data</p>
    </div>
""", unsafe_allow_html=True)


# ── Load Data from Database ──────────────────────────────────
# get_all_sales() queries the SQLite daily_sales table and
# returns all records as a pandas DataFrame sorted by date.
# This includes the original tbats_train.csv data AND any
# new days added by the user through the Daily Update page.
df = get_all_sales()

# If no data found, show a warning and stop the page
if df.empty:
    st.warning("⚠️ No data found in database. Please check your setup.")
    st.stop()

# ── Feature Engineering for Dashboard ───────────────────────
# Extract time components from the date column for grouping.
# These are added as new columns to the DataFrame.
df['year']       = df['date'].dt.year    # e.g., 2019, 2020, 2021
df['month']      = df['date'].dt.month   # e.g., 1, 2, ... 12
df['month_name'] = df['date'].dt.strftime('%b')  # e.g., Jan, Feb, Mar


# ── Summary Metrics Row ──────────────────────────────────────
# Display 4 key metrics at the top of the dashboard in a
# 4-column layout so users see the most important numbers
# at a glance without needing to scroll.
st.subheader("📌 Key Metrics")

m1, m2, m3, m4 = st.columns(4)

with m1:
    # Total number of days of data available in the database
    st.metric("📅 Total Days", f"{len(df):,}")

with m2:
    # Total sales across all days (formatted with commas)
    total_sales = df['sales'].sum()
    st.metric("💰 Total Sales", f"{total_sales:,.0f}")

with m3:
    # Average daily sales across the entire historical period
    avg_sales = df['sales'].mean()
    st.metric("📈 Avg Daily Sales", f"{avg_sales:,.0f}")

with m4:
    # Most recent date available in the database.
    # get_latest_date() queries MAX(date) from daily_sales.
    latest = get_latest_date()
    st.metric("🗓️ Latest Date", str(latest)[:10])

st.divider()


# ── Full Historical Sales Trend ──────────────────────────────
st.subheader("📈 Historical Sales Trend")

# px.line() creates an interactive line chart.
# x='date'  → x-axis shows dates (datetime objects)
# y='sales' → y-axis shows total daily sales values
# color_discrete_sequence → sets the line color to blue
fig_trend = px.line(
    df,
    x='date',
    y='sales',
    title='Daily Total Sales — Full Historical Period',
    labels={'sales': 'Total Sales', 'date': 'Date'},
    color_discrete_sequence=['#3b82f6']  # blue line color
)

# Update chart layout for dark professional theme.
# plot_bgcolor  → background color inside the chart area
# paper_bgcolor → background color outside the chart area
# font_color    → color for all axis labels and tick text
# hovermode='x unified' → shows all values on hover at x
fig_trend.update_layout(
    plot_bgcolor='#0f172a',
    paper_bgcolor='#0f172a',
    font_color='#94a3b8',
    title_font_size=16,
    hovermode='x unified',
    xaxis=dict(showgrid=False),  # clean look — hide x gridlines
    yaxis=dict(showgrid=True, gridcolor='#1e293b')  # subtle y gridlines
)

# st.plotly_chart() renders the plotly figure inline.
# use_container_width=True makes the chart fill full page width.
st.plotly_chart(fig_trend, use_container_width=True)


# ── Monthly Sales Bar Chart ──────────────────────────────────
st.subheader("📅 Monthly Sales Breakdown")

# Group data by year and month, then sum sales for each group.
# reset_index() converts groupby result back to a regular DataFrame
# so plotly can access columns by name.
monthly_df = df.groupby(['year', 'month', 'month_name'])['sales'].sum().reset_index()

# Sort by year and month to ensure bars appear in calendar order
monthly_df = monthly_df.sort_values(['year', 'month'])

# Create a grouped bar chart: x = month name, bars grouped by year.
# color='year' assigns a different color to each year automatically.
# barmode='group' places bars side by side (not stacked).
fig_monthly = px.bar(
    monthly_df,
    x='month_name',
    y='sales',
    color='year',
    barmode='group',
    title='Monthly Sales by Year',
    labels={'sales': 'Total Sales', 'month_name': 'Month', 'year': 'Year'},
    color_discrete_sequence=px.colors.qualitative.Set2  # distinct colors per year
)

fig_monthly.update_layout(
    plot_bgcolor='#0f172a',
    paper_bgcolor='#0f172a',
    font_color='#94a3b8',
    title_font_size=16,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='#1e293b')
)

st.plotly_chart(fig_monthly, use_container_width=True)


# ── Year-over-Year Comparison ────────────────────────────────
st.subheader("📊 Year-over-Year Sales Comparison")

# Group by year and sum total annual sales
yearly_df = df.groupby('year')['sales'].sum().reset_index()

# pct_change() computes: (current - previous) / previous * 100
# This gives the % growth/decline compared to the prior year.
# The first year will be NaN (no previous year to compare).
yearly_df['yoy_change'] = yearly_df['sales'].pct_change() * 100

# Create a 2-column layout: bar chart on left, table on right
left, right = st.columns(2)

with left:
    # Bar chart showing total annual sales per year.
    # color='sales' applies a gradient — taller bars are darker.
    # color_continuous_scale='Blues' → blue gradient scale.
    fig_yearly = px.bar(
        yearly_df,
        x='year',
        y='sales',
        title='Total Annual Sales by Year',
        labels={'sales': 'Total Sales', 'year': 'Year'},
        color='sales',
        color_continuous_scale='Blues'
    )
    fig_yearly.update_layout(
        plot_bgcolor='#0f172a',
        paper_bgcolor='#0f172a',
        font_color='#94a3b8',
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#1e293b')
    )
    st.plotly_chart(fig_yearly, use_container_width=True)

with right:
    # Display the yearly summary as a formatted table
    st.markdown("**Annual Sales Summary**")

    # Format the DataFrame for readable display
    display_df = yearly_df.copy()
    # Apply comma formatting to the total sales column
    display_df['sales'] = display_df['sales'].apply(lambda x: f"{x:,.0f}")
    # Apply +/- percentage formatting; show '—' for the first year (NaN)
    display_df['yoy_change'] = display_df['yoy_change'].apply(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"
    )
    display_df.columns = ['Year', 'Total Sales', 'YoY Change']

    # st.dataframe() renders an interactive scrollable table.
    # hide_index=True removes the pandas row number column.
    st.dataframe(display_df, hide_index=True, use_container_width=True)

st.divider()


# ── Rolling 30-Day Average Trend ────────────────────────────
# A rolling average smooths out day-to-day noise so the
# underlying trend is easier to see in the chart.
st.subheader("📉 30-Day Rolling Average Sales")

# Calculate 30-day rolling mean — each point is the average
# of the last 30 days. min_periods=1 allows calculation even
# before 30 days of data are available (avoids NaN at start).
df_roll = df.sort_values('date').copy()
df_roll['rolling_avg'] = df_roll['sales'].rolling(window=30, min_periods=1).mean()

# Create a figure with two traces:
# Trace 1 (blue, faint) → raw daily sales
# Trace 2 (orange)      → 30-day smoothed rolling average
fig_roll = go.Figure()

fig_roll.add_trace(go.Scatter(
    x=df_roll['date'],
    y=df_roll['sales'],
    mode='lines',
    name='Daily Sales',
    line=dict(color='#3b82f6', width=1, dash='dot'),  # thin dotted blue
    opacity=0.4   # make it faint so the rolling average stands out
))

fig_roll.add_trace(go.Scatter(
    x=df_roll['date'],
    y=df_roll['rolling_avg'],
    mode='lines',
    name='30-Day Rolling Avg',
    line=dict(color='#f59e0b', width=2.5)  # solid orange, bolder
))

fig_roll.update_layout(
    title='Raw Sales vs 30-Day Rolling Average',
    xaxis_title='Date',
    yaxis_title='Sales',
    plot_bgcolor='#0f172a',
    paper_bgcolor='#0f172a',
    font_color='#94a3b8',
    title_font_size=16,
    hovermode='x unified',
    legend=dict(bgcolor='#1e293b', bordercolor='#334155', borderwidth=1),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='#1e293b')
)

st.plotly_chart(fig_roll, use_container_width=True)

st.divider()


# ── Model Performance Leaderboard ───────────────────────────
# Shows evaluation results for all 5 models trained in the notebook.
# Rankings are based on RMSE (lower = better) and R² (higher = better).
# Note: LSTM ranked last due to MinMaxScaler leakage before the train/
# test split — the scaler saw test min/max, inflating its error to ~6.7B.
st.subheader("🏆 Model Performance Leaderboard")
st.caption(
    "All models evaluated on the held-out test set (last 20% of data, chronological split). "
    "LSTM result reflects a known scaler leakage issue — fix covered in the notebook."
)

lb_col1, lb_col2 = st.columns([3, 2])

with lb_col1:
    # Horizontal bar chart — RMSE comparison (best model at top).
    # LSTM's RMSE is ~6.7 billion so we cap the bar visually and show
    # the actual value as a text label outside the bar.
    DISPLAY_CAP = 30_000

    lb_models = [f"{r['medal']} {r['model']}" for r in MODEL_RANKINGS]
    lb_rmse   = [min(r['rmse'], DISPLAY_CAP) for r in MODEL_RANKINGS]
    lb_actual = [
        f"RMSE: {r['rmse']:,.2f}" if r['rmse'] < 1_000_000
        else f"RMSE: {r['rmse']:.2e}"
        for r in MODEL_RANKINGS
    ]
    lb_colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#6b7280']

    fig_lb = go.Figure(go.Bar(
        x=lb_rmse[::-1],
        y=lb_models[::-1],
        orientation='h',
        marker_color=lb_colors[::-1],
        text=lb_actual[::-1],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11),
        cliponaxis=False
    ))

    fig_lb.update_layout(
        title='RMSE Comparison — lower is better (LSTM bar capped for scale)',
        xaxis_title='RMSE',
        plot_bgcolor='#0f172a',
        paper_bgcolor='#0f172a',
        font_color='#94a3b8',
        title_font_size=13,
        xaxis=dict(showgrid=True, gridcolor='#1e293b', range=[0, DISPLAY_CAP * 1.5]),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=160, t=50, b=30),
        height=320
    )

    st.plotly_chart(fig_lb, use_container_width=True)

with lb_col2:
    st.markdown("**Full Rankings**")

    lb_df = pd.DataFrame([
        {
            "Rank":  f"{r['medal']} #{r['rank']}",
            "Model": r['model'],
            "RMSE":  f"{r['rmse']:,.2f}" if r['rmse'] < 1_000_000 else f"{r['rmse']:.2e}",
            "MSE":   f"{r['mse']:,.2f}"  if r['mse']  < 1_000_000 else f"{r['mse']:.2e}",
            "R²":    f"{r['r2']:+.4f}",
        }
        for r in MODEL_RANKINGS
    ])

    st.dataframe(lb_df, hide_index=True, use_container_width=True)

    st.markdown("""
    <div style='background:#0f2027;border-left:3px solid #10b981;padding:0.8rem 1rem;
    border-radius:6px;margin-top:0.8rem;font-size:0.82rem;color:#94a3b8;line-height:1.6;'>
    ✅ <b style='color:#ecfdf5'>TBATS selected</b> as production model.<br>
    R² = 0.9951 · RMSE = 1,405.30<br><br>
    ⚠️ <b style='color:#fbbf24'>LSTM</b> result is inflated by a scaler
    leakage bug (fit on full dataset before split).
    After fixing, LSTM is expected to rank 2nd or 3rd.
    </div>
    """, unsafe_allow_html=True)
