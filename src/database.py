# ============================================================
# src/database.py — Database Handler
# ============================================================
# This file handles all interactions with the SQLite database.
# It is responsible for:
#   1. Creating the database and tables on first run
#   2. Loading TBATS training data into the database
#   3. Reading data from the database for forecasting/dashboard
#   4. Appending new daily data as it comes in from the UI
#
# ✅ Updated for TBATS model:
#    - Only 2 columns are needed: date + sales
#    - No simulation CSV — user manually enters new day data
#    - No store-level or feature tables required
# ============================================================

import sqlite3
# sqlite3 → Python's built-in module for working with SQLite
# databases. SQLite stores the entire database in a single
# .db file on disk. No server or installation needed.
# Perfect for capstone and small-to-medium applications.

import pandas as pd
# pandas → powerful data manipulation library.
# Used here to read CSV files and convert database query
# results into DataFrames for easy processing.

import os
# os → built-in module for file and folder operations.
# Used to check if the database file already exists.

import sys
# sys → built-in module for system-level operations.
# Used here to add the parent folder to Python's search path
# so we can import config.py from the root project folder.

# Add the parent directory to path so we can import config.py
# sys.path.insert(0, ...) tells Python to look in the parent
# folder when searching for modules to import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all path constants defined in config.py
from config import DB_PATH, TBATS_TRAIN_PATH


# ── Function 1: Initialize Database ─────────────────────────
def init_database():
    """
    Creates the SQLite database and required tables if they
    don't already exist. Also loads the TBATS training data
    (tbats_train.csv) into the database on the very first run.

    The daily_sales table stores 2 columns:
        - date  (TEXT, PRIMARY KEY) — e.g. '2019-01-01'
        - sales (REAL)              — total daily sales
    """

    # sqlite3.connect() opens a connection to the database.
    # If the .db file doesn't exist, SQLite creates it automatically.
    conn = sqlite3.connect(DB_PATH)

    # conn.cursor() creates a cursor object which is used to
    # execute SQL commands (CREATE TABLE, INSERT, SELECT, etc.)
    cursor = conn.cursor()

    # ── Create daily_sales table ─────────────────────────────
    # This SQL command creates the table only if it doesn't
    # exist yet. The table stores daily aggregated sales data.
    # Each row = one day's total sales across all stores.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_sales (
            date  TEXT PRIMARY KEY,
            sales REAL
        )
    """)
    # TEXT = string data type in SQLite (stores dates as strings)
    # REAL = floating point number (stores sales values)
    # PRIMARY KEY = ensures no duplicate dates are inserted

    # Save (commit) the table creation command to disk
    conn.commit()

    # ── Check if table already has data ──────────────────────
    # We only load tbats_train.csv if the table is empty.
    # This prevents overwriting data added by the daily pipeline.
    cursor.execute("SELECT COUNT(*) FROM daily_sales")
    row_count = cursor.fetchone()[0]

    # ── Load TBATS training data on first run ─────────────────
    # If the table is empty, read tbats_train.csv and insert
    # all records into the database. tbats_train.csv has exactly
    # 2 columns: date (index) and sales (total daily sales).
    if row_count == 0:
        # Read the TBATS training CSV file into a DataFrame.
        # index_col=0 → use the first column (date) as the index
        # parse_dates=True → automatically parse date strings
        tbats_df = pd.read_csv(TBATS_TRAIN_PATH, index_col=0, parse_dates=True)

        # Reset index to make date a regular column so we can
        # rename it and insert it into the database cleanly
        tbats_df = tbats_df.reset_index()

        # Ensure column names are exactly ['date', 'sales']
        # regardless of what they were named in the CSV file
        tbats_df.columns = ['date', 'sales']

        # Convert date to string format (YYYY-MM-DD) for SQLite
        # SQLite has no native DATE type — TEXT is standard
        tbats_df['date'] = tbats_df['date'].astype(str).str[:10]

        # to_sql() writes the DataFrame rows into the SQLite table.
        # if_exists='append' adds rows without dropping the table.
        # index=False prevents pandas from writing row numbers.
        tbats_df.to_sql('daily_sales', conn, if_exists='append', index=False)

        print(f"✅ Database initialized with {len(tbats_df)} TBATS training records")

    # Close the database connection to free up resources
    conn.close()


# ── Function 2: Get All Sales Data ──────────────────────────
def get_all_sales():
    """
    Retrieves all daily sales records from the database
    and returns them as a pandas DataFrame sorted by date.

    Returns:
        DataFrame with columns: ['date', 'sales']
        date column is converted to datetime type.
    """

    # Open database connection
    conn = sqlite3.connect(DB_PATH)

    # pd.read_sql_query() runs a SQL SELECT query and directly
    # returns the result as a pandas DataFrame.
    # ORDER BY date → sorted oldest to newest (ascending)
    df = pd.read_sql_query(
        "SELECT * FROM daily_sales ORDER BY date",
        conn
    )

    # Close connection after reading
    conn.close()

    # Convert the date column from string back to datetime
    # so we can do date-based operations and chart plotting
    df['date'] = pd.to_datetime(df['date'])

    return df


# ── Function 3: Get Latest Date ─────────────────────────────
def get_latest_date():
    """
    Returns the most recent date available in the database.
    Used to display 'Forecasting from: YYYY-MM-DD' in the UI
    and to validate new dates entered by the user.

    Returns:
        str: Date string in 'YYYY-MM-DD' format, e.g. '2022-08-15'
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # MAX(date) returns the latest date string in the table.
    # Since dates are stored as YYYY-MM-DD text, lexicographic
    # order matches chronological order — MAX works correctly.
    cursor.execute("SELECT MAX(date) FROM daily_sales")

    # fetchone() retrieves a single row from the result.
    # [0] gets the first (and only) column value from that row.
    latest_date = cursor.fetchone()[0]

    conn.close()

    return latest_date


# ── Function 4: Append New Daily Data ───────────────────────
def append_new_day(date, sales):
    """
    Adds a new day's sales record into the database.
    Called by the Daily Update page when the user manually
    enters a new date and its actual sales value.

    Parameters:
        date  (str):   Date string in 'YYYY-MM-DD' format
        sales (float): Total daily sales value for that day

    Notes:
        - INSERT OR IGNORE prevents duplicate date entries
        - The ? placeholders prevent SQL injection attacks
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # INSERT OR IGNORE means: if this date already exists
    # in the table (PRIMARY KEY conflict), skip the insert.
    # This prevents duplicate entries for the same date.
    cursor.execute("""
        INSERT OR REPLACE INTO daily_sales (date, sales)
        VALUES (?, ?)
    """, (str(date)[:10], float(sales)))
    # str(date)[:10] → ensures format is 'YYYY-MM-DD'
    # float(sales)   → ensures sales is a numeric value

    # Save the new row to disk
    conn.commit()
    conn.close()


# ── Function 5: Check If Date Exists ────────────────────────
def date_exists(date):
    """
    Checks whether a given date already exists in the database.
    Used by the Daily Update page to prevent the user from
    entering duplicate dates.

    Parameters:
        date (str): Date string in 'YYYY-MM-DD' format

    Returns:
        bool: True if the date already exists, False otherwise
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM daily_sales WHERE date = ?",
        (str(date)[:10],)
    )

    count = cursor.fetchone()[0]
    conn.close()

    # count > 0 means the date already has a record in the DB
    return count > 0
