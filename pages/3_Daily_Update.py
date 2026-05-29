# ============================================================
# pages/3_Daily_Update.py — Daily Data Entry Page
# ============================================================
# This page allows the user to manually add new daily sales
# records to the SQLite database, extending the training data
# beyond the original tbats_train.csv date range.
#
# In production retail systems, sales data flows in every day
# from Point-of-Sale (POS) systems. This page simulates that
# pipeline by letting the user type in the next date and its
# corresponding actual total sales figure.
#
# Each submission:
#   1. Validates the date (must be after the current latest date
#      OR an existing date — in which case it will be overwritten)
#   2. Validates the sales value (must be a positive number)
#   3. Upserts the record into the SQLite database
#      (INSERT OR REPLACE — overwrites if date already exists)
#   4. Shows the updated sales chart and last 10 records
#
# ✅ Updated for TBATS:
#    - No simulation CSV — user enters real date + sales manually
#    - No simulation index tracking needed
#    - Existing dates are OVERWRITTEN (upsert behaviour)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
from datetime import timedelta

# Add root folder to Python's search path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database functions needed for daily update operations
from src.database import (
    get_all_sales,      # get all records from the database
    get_latest_date,    # get the most recent date in the database
    append_new_day,     # insert/update a daily record in the database
    date_exists         # check if a date is already in the database
)

# Import app settings
from config import APP_TITLE

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title=f"Daily Update | {APP_TITLE}",
    page_icon="📅",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .page-header {
            background: linear-gradient(135deg, #064e3b, #065f46, #047857);
            padding: 2rem; border-radius: 14px;
            margin-bottom: 1.5rem; text-align: center;
        }
        .page-header h2 { color: #ecfdf5; font-size: 1.9rem; margin: 0; }
        .page-header p  { color: #a7f3d0; margin-top: 0.3rem; }
        .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ── Page Header ──────────────────────────────────────────────
st.markdown("""
    <div class="page-header">
        <h2>📅 Daily Data Entry</h2>
        <p>Add or update daily sales records in the database</p>
    </div>
""", unsafe_allow_html=True)

# ── How It Works Explanation ─────────────────────────────────
with st.expander("ℹ️ How does this work?", expanded=False):
    st.markdown("""
    **In a real production system:**
    - Every night, store POS systems send that day's sales data to a central server
    - The server validates and stores the data in the database
    - The forecasting model then uses the updated database for fresh predictions

    **In this simulation:**
    - You manually enter **any date** along with the **actual total sales** for that day
    - If the date **already exists**, its sales value will be **overwritten** (upsert)
    - If the date is **new**, it will be inserted as a fresh record
    - The record is saved into the SQLite database immediately

    **After adding data:**
    - Go to **🔮 Forecast** to generate predictions based on the updated history
    - Go to **🔁 Retrain** to retrain the TBATS model with the new data included
    """)

st.divider()

# ── Current Database Status ──────────────────────────────────
st.subheader("📊 Current Database Status")

latest_date_str = get_latest_date()
latest_date     = pd.to_datetime(latest_date_str)
current_df      = get_all_sales()

next_expected_date = latest_date + timedelta(days=1)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("🗓️ Latest Date in DB", str(latest_date)[:10])

with c2:
    st.metric("📋 Total Records", f"{len(current_df):,}")

with c3:
    st.metric("➡️ Next Expected Date", str(next_expected_date)[:10])

st.divider()

# ── Data Entry Form ──────────────────────────────────────────
st.subheader("✍️ Enter New Daily Sales Data")

inp_col1, inp_col2 = st.columns(2)

with inp_col1:
    selected_date = st.date_input(
        "📅 Select Date",
        value=next_expected_date.date(),
        help="Select the date for the sales record. "
             "If the date already exists in the database, its value will be overwritten."
    )

with inp_col2:
    sales_value = st.number_input(
        "💰 Enter Total Sales",
        min_value=0.0,
        step=1.0,
        format="%.2f",
        help="Enter the total actual sales for this date (sum across all stores)."
    )

# ── Live hint: warn user if selected date will overwrite ─────
selected_dt = pd.to_datetime(selected_date)
if date_exists(str(selected_date)):
    st.info(
        f"ℹ️ **{selected_date}** already exists in the database. "
        f"Submitting will **overwrite** the existing sales value for this date."
    )

st.markdown("")  # spacing

if st.button("📥 Add / Update Database", use_container_width=True, type="primary"):

    # ── Validation: Sales must be greater than 0 ──────────────
    if sales_value <= 0:
        st.error("❌ Sales value must be greater than 0. Please enter a valid amount.")

    else:
        # ── Upsert into database ──────────────────────────────
        # append_new_day() should use INSERT OR REPLACE INTO daily_sales
        # so that existing dates are overwritten, not skipped.
        is_update = date_exists(str(selected_date))

        with st.spinner(f"{'Updating' if is_update else 'Adding'} {selected_date} in database..."):
            append_new_day(str(selected_date), sales_value)

        if is_update:
            st.success(
                f"🔄 Successfully updated: **{selected_date}** — "
                f"Total Sales overwritten to: **{sales_value:,.2f}**"
            )
        else:
            st.success(
                f"✅ Successfully added: **{selected_date}** — "
                f"Total Sales: **{sales_value:,.2f}**"
            )

        st.rerun()

st.divider()

# ── Updated Sales Chart ──────────────────────────────────────
st.subheader("📈 Current Sales History")

all_df = get_all_sales()

if not all_df.empty:
    fig = px.line(
        all_df,
        x='date',
        y='sales',
        title='All Sales Records in Database (Including Newly Added / Updated Days)',
        labels={'sales': 'Total Sales', 'date': 'Date'},
        color_discrete_sequence=['#10b981']
    )

    fig.update_layout(
        plot_bgcolor='#0f172a',
        paper_bgcolor='#0f172a',
        font_color='#94a3b8',
        hovermode='x unified',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#1e293b')
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗃️ Last 10 Records in Database")

    last10 = all_df.tail(10).copy()
    last10['date']  = last10['date'].astype(str).str[:10]
    last10['sales'] = last10['sales'].apply(lambda x: f"{x:,.2f}")
    last10.columns  = ['Date', 'Total Sales']

    st.dataframe(last10, hide_index=True, use_container_width=True)
