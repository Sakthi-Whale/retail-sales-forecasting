# ============================================================
# pages/5_Delete_Record.py — Delete a Sales Record by Date
# ============================================================
# This page allows the user to delete a specific daily sales
# record from the SQLite database by selecting its date.
#
# Flow:
#   1. User picks a date from the date picker
#   2. App checks if that date exists in the database
#   3. If found → shows a preview of the record (date + sales)
#   4. User must tick a confirmation checkbox to unlock delete
#   5. On confirm → record is deleted from the database
#   6. Page refreshes and shows updated table + chart
#
# Safety guards:
#   - Date existence check before showing delete option
#   - Confirmation checkbox prevents accidental deletion
#   - Clear success / warning / error feedback at every step
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import (
    get_all_sales,
    date_exists,
)
from config import APP_TITLE

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title=f"Delete Record | {APP_TITLE}",
    page_icon="🗑️",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .page-header {
            background: linear-gradient(135deg, #7f1d1d, #991b1b, #b91c1c);
            padding: 2rem; border-radius: 14px;
            margin-bottom: 1.5rem; text-align: center;
        }
        .page-header h2 { color: #fff1f2; font-size: 1.9rem; margin: 0; }
        .page-header p  { color: #fecaca; margin-top: 0.3rem; }
        .block-container { padding-top: 1rem; }
        .preview-box {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
        }
        .preview-box h4 { color: #f8fafc; margin: 0 0 0.5rem 0; }
        .preview-box p  { color: #94a3b8; margin: 0; font-size: 0.95rem; }
    </style>
""", unsafe_allow_html=True)

# ── Page Header ──────────────────────────────────────────────
st.markdown("""
    <div class="page-header">
        <h2>🗑️ Delete Sales Record</h2>
        <p>Permanently remove a daily sales entry from the database by date</p>
    </div>
""", unsafe_allow_html=True)

st.warning(
    "⚠️ **Caution:** Deleting a record is permanent and cannot be undone. "
    "Make sure you select the correct date before confirming.",
    icon="⚠️"
)

st.divider()

# ── Helper: delete function (inline, no extra DB import needed) ──
import sqlite3
from src.database import DB_PATH   # reuse the same DB path constant

def delete_record_by_date(date_str: str) -> bool:
    """
    Delete the row with the given date from daily_sales.
    Returns True if a row was actually deleted, False otherwise.
    """
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM daily_sales WHERE date = ?",
            (date_str,)
        )
        deleted = cursor.rowcount  # number of rows affected
        conn.commit()
        conn.close()
        return deleted > 0
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def get_record_by_date(date_str: str):
    """
    Fetch the sales value for a specific date.
    Returns the sales float or None if not found.
    """
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sales FROM daily_sales WHERE date = ?",
            (date_str,)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None

# ── Current Database Status ──────────────────────────────────
st.subheader("📊 Current Database Status")

all_df = get_all_sales()

c1, c2 = st.columns(2)
with c1:
    st.metric("📋 Total Records", f"{len(all_df):,}")
with c2:
    if not all_df.empty:
        st.metric("🗓️ Date Range",
                  f"{str(all_df['date'].min())[:10]}  →  {str(all_df['date'].max())[:10]}")

st.divider()

# ── Delete Form ──────────────────────────────────────────────
st.subheader("🗓️ Select Date to Delete")

# Build a list of all existing dates for the picker's min/max bounds
if all_df.empty:
    st.info("ℹ️ The database has no records to delete.")
    st.stop()

all_dates      = pd.to_datetime(all_df['date'])
min_date       = all_dates.min().date()
max_date       = all_dates.max().date()

selected_date  = st.date_input(
    "Pick a date",
    value=max_date,          # default to latest record
    min_value=min_date,
    max_value=max_date,
    help="Choose the date of the record you want to delete."
)

date_str = str(selected_date)

# ── Record Preview ───────────────────────────────────────────
st.markdown("")

if date_exists(date_str):
    sales_val = get_record_by_date(date_str)

    # Show a preview card of what will be deleted
    st.markdown(f"""
        <div class="preview-box">
            <h4>📄 Record Found</h4>
            <p>
                <strong style="color:#f8fafc;">Date:</strong>
                <span style="color:#fca5a5;">&nbsp;{date_str}</span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <strong style="color:#f8fafc;">Total Sales:</strong>
                <span style="color:#fca5a5;">&nbsp;{sales_val:,.2f}</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Confirmation Checkbox ─────────────────────────────────
    confirmed = st.checkbox(
        f"✅ Yes, I want to permanently delete the record for **{date_str}**",
        value=False
    )

    # Delete button — disabled until checkbox is ticked
    delete_clicked = st.button(
        "🗑️ Delete Record",
        use_container_width=True,
        type="primary",
        disabled=not confirmed
    )

    if delete_clicked and confirmed:
        with st.spinner(f"Deleting record for {date_str}..."):
            success = delete_record_by_date(date_str)

        if success:
            st.success(f"✅ Record for **{date_str}** (Sales: {sales_val:,.2f}) has been permanently deleted.")
            st.rerun()
        else:
            st.error("❌ Something went wrong. The record could not be deleted. Please try again.")

else:
    # Date selected but not in database
    st.info(
        f"ℹ️ No record found for **{date_str}**. "
        f"Please select a date that exists in the database."
    )

st.divider()

# ── Current Sales Chart ──────────────────────────────────────
st.subheader("📈 Current Sales History")

if not all_df.empty:
    fig = px.line(
        all_df,
        x='date',
        y='sales',
        title='All Sales Records in Database',
        labels={'sales': 'Total Sales', 'date': 'Date'},
        color_discrete_sequence=['#ef4444']   # red line to match delete theme
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

    # ── Full Records Table ────────────────────────────────────
    st.subheader("🗃️ All Records in Database")

    display_df          = all_df.copy()
    display_df['date']  = display_df['date'].astype(str).str[:10]
    display_df['sales'] = display_df['sales'].apply(lambda x: f"{x:,.2f}")
    display_df.columns  = ['Date', 'Total Sales']

    st.dataframe(display_df, hide_index=True, use_container_width=True)
