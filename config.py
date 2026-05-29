# ============================================================
# config.py — Central Configuration File
# ============================================================
# This file stores all the important paths and settings used
# across the entire application. Instead of hardcoding paths
# in every file, we define them here once and import them
# wherever needed. This makes it easy to change paths in one
# place without touching every other file.
#
# ✅ Updated for TBATS model — no store-level feature files
#    needed. TBATS works directly on date + total daily sales.
# ============================================================

import os
# os → Python's built-in module for interacting with the
# operating system. Used here to build file paths that work
# on both Windows and Mac/Linux automatically.

# ── Base Directory ──────────────────────────────────────────
# os.path.dirname(__file__) gives the folder where this
# config.py file is located. This becomes our base reference
# point for all other paths below.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# os.path.abspath() converts a relative path to an absolute
# full path (e.g., C:/Users/user/retail_forecasting)

# ── Model Paths ─────────────────────────────────────────────
# This is the folder where your trained models are saved.
# os.path.join() combines folder names in a way that works
# on both Windows (\) and Linux/Mac (/) automatically.
MODELS_DIR = os.path.join(BASE_DIR, "models")

# This is the folder where your processed CSV files are saved.
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

# Path to the saved TBATS model file (.pkl format)
# .pkl is a pickle file — Python's way of saving any object
# to disk so it can be loaded back later without retraining.
# TBATS model is saved/loaded using joblib.
TBATS_MODEL_PATH = os.path.join(MODELS_DIR, "tbats_model.pkl")

# ── Data Paths ───────────────────────────────────────────────
# This is the folder where your processed CSV files are saved.
DATA_DIR = os.path.join(BASE_DIR, "processed")

# TBATS training data — a simple 2-column CSV (date, sales).
# This is the aggregated daily total sales across all stores
# that the TBATS model was trained on during the notebook.
# Format: date (YYYY-MM-DD), sales (integer/float)
TBATS_TRAIN_PATH = os.path.join(PROCESSED_DIR, "tbats_train_data.csv")

# ── Database Path ────────────────────────────────────────────
# SQLite database file path. SQLite is a lightweight database
# that stores everything in a single .db file — no server
# setup needed, perfect for this project.
DB_PATH = os.path.join(BASE_DIR, "forecasting.db")

# Page title shown in the browser tab and app header
APP_TITLE = "Retail Sales Forecasting Dashboard"

# Emoji icon shown in the browser tab next to the page title
APP_ICON = "🛒"
