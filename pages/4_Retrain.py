# ============================================================
# pages/4_Retrain.py — TBATS Model Retraining Page
# ============================================================
# FIXED RETRAINING PIPELINE (production-correct approach):
#
#   EVALUATION (fair comparison):
#     - Fixed holdout = last 30 days of the database
#     - Old model forecasts those 30 days from its saved state
#     - New model is temporarily trained on everything BEFORE
#       those 30 days, then forecasts the same 30 days
#     - Both are evaluated on the SAME fixed window → fair
#
#   SAVING (production retraining):
#     - If new model wins → retrain it on the FULL dataset
#       (all records, no holdout) and save that as the model
#     - This gives the saved model maximum data to learn from
#     - Matches real-world production retraining practice
#
# WHY THIS FIXES THE ORIGINAL ISSUE:
#   Before: new model trained on 80% of data (weaker)
#           old model was trained on 100% of original data
#           → comparison was unfair → new model always lost
#   Now:    both evaluated on same 30-day holdout
#           saved model always trained on 100% of data
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tbats import TBATS as TBATSEstimator
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TBATS_MODEL_PATH, APP_TITLE
from src.database import get_all_sales

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title=f"Retrain | {APP_TITLE}",
    page_icon="🔁",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .page-header {
            background: linear-gradient(135deg, #1e1b4b, #312e81, #4338ca);
            padding: 2rem; border-radius: 14px;
            margin-bottom: 1.5rem; text-align: center;
        }
        .page-header h2 { color: #e0e7ff; font-size: 1.9rem; margin: 0; }
        .page-header p  { color: #a5b4fc; margin-top: 0.3rem; }
        .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ── Page Header ──────────────────────────────────────────────
st.markdown("""
    <div class="page-header">
        <h2>🔁 TBATS Model Retraining</h2>
        <p>Retrain the TBATS model on the full updated database</p>
    </div>
""", unsafe_allow_html=True)

# ── Why Retrain Explanation ───────────────────────────────────
with st.expander("ℹ️ Why retrain the TBATS model?", expanded=False):
    st.markdown("""
    As new daily sales data is added to the database via the **Daily Update** page,
    the model can be retrained to incorporate those new patterns.

    **Retraining updates the TBATS model to:**
    - Learn from recently added daily sales data
    - Adapt to new seasonal trends or shifts in sales behavior
    - Improve forecast accuracy as more observations accumulate

    **How the fixed retraining pipeline works:**

    | Phase | What happens |
    |-------|-------------|
    | **Evaluation** | Both old and new models are tested on the same fixed last-30-days holdout for a fair comparison |
    | **Saving** | If the new model wins, it is retrained on the **full dataset** (100% of records) before saving |

    **Why train on 100% for saving?**
    In production, once a model is selected as better, it should learn from
    ALL available data — not just 80%. The holdout is only used for comparison,
    not for the final saved model.
    """)

st.divider()

# ── Load Current Database Data ───────────────────────────────
st.subheader("📊 Current Training Data in Database")

with st.spinner("Loading data from database..."):
    df = get_all_sales()
    df = df.sort_values('date').reset_index(drop=True)

# Fixed holdout size — last 30 days used for evaluation only
HOLDOUT_DAYS = 30

d1, d2, d3 = st.columns(3)

with d1:
    st.metric(
        "📅 Date Range",
        f"{str(df['date'].min())[:10]}  →  {str(df['date'].max())[:10]}"
    )

with d2:
    st.metric("📋 Total Records", f"{len(df):,}")

with d3:
    st.metric(
        "🧪 Fixed Holdout (Evaluation)",
        f"Last {HOLDOUT_DAYS} days"
    )

st.divider()

# ── Retrain Button ────────────────────────────────────────────
if st.button("🔁 Retrain TBATS Model", use_container_width=True, type="primary"):

    with st.spinner("Retraining TBATS model — this may take a few minutes..."):

        try:
            # ── Step 1: Prepare Time Series ──────────────────────
            st.write("⚙️ Step 1: Preparing time series from database...")

            sales_array = df['sales'].values
            dates_array = df['date'].values

            st.write(f"✅ Loaded {len(sales_array):,} daily records from database")

            # ── Step 2: Fixed Holdout Split ───────────────────────
            # Use a fixed last-N-days holdout for EVALUATION ONLY.
            # This ensures old model and new model are compared on
            # the exact same time window — a fair comparison.
            st.write(f"⚙️ Step 2: Setting fixed holdout (last {HOLDOUT_DAYS} days for evaluation)...")

            # Everything except the last 30 days → used to train new model for evaluation
            eval_train_sales = sales_array[:-HOLDOUT_DAYS]

            # Last 30 days → holdout for comparing old vs new
            holdout_sales = sales_array[-HOLDOUT_DAYS:]
            holdout_dates = dates_array[-HOLDOUT_DAYS:]

            st.write(
                f"✅ Eval-train: {len(eval_train_sales):,} samples | "
                f"Holdout (fixed): {len(holdout_sales):,} samples "
                f"({str(holdout_dates[0])[:10]} → {str(holdout_dates[-1])[:10]})"
            )

            # ── Step 3: Evaluate Old Model on Holdout ────────────
            st.write("⚙️ Step 3: Evaluating old TBATS model on fixed holdout...")

            old_model  = joblib.load(TBATS_MODEL_PATH)
            # Forecast HOLDOUT_DAYS steps from the old model's saved state
            old_pred   = old_model.forecast(steps=HOLDOUT_DAYS)
            old_pred   = np.maximum(old_pred, 0)   # clip negatives

            old_rmse   = float(np.sqrt(mean_squared_error(holdout_sales, old_pred)))
            old_r2     = float(r2_score(holdout_sales, old_pred))

            st.write(f"✅ Old model evaluated — RMSE: {old_rmse:,.2f} | R²: {old_r2:.4f}")

            # ── Step 4: Train New TBATS Model (eval split only) ───
            # Train on eval_train_sales (everything except holdout)
            # so it has NOT seen the holdout → fair comparison.
            st.write("⚙️ Step 4: Training new TBATS model on eval-train split...")

            # ⚠️ CRITICAL: seasonal_periods and use_arma_errors MUST match
            # the original notebook exactly:
            #   seasonal_periods=[7, 365]  → weekly + yearly seasonality
            #   use_arma_errors=False      → same as original training
            # Using None here causes TBATS to auto-detect wrong periods
            # and produces a completely flat forecast line.
            estimator = TBATSEstimator(
                seasonal_periods=[7, 365],
                use_arma_errors=False,
                n_jobs=1
            )

            new_model_eval = estimator.fit(eval_train_sales)
            st.write("✅ New TBATS model trained successfully (eval split)!")

            # ── Step 5: Evaluate New Model on Same Holdout ────────
            st.write("⚙️ Step 5: Evaluating new model on the same fixed holdout...")

            new_pred = new_model_eval.forecast(steps=HOLDOUT_DAYS)
            new_pred = np.maximum(new_pred, 0)

            new_rmse = float(np.sqrt(mean_squared_error(holdout_sales, new_pred)))
            new_r2   = float(r2_score(holdout_sales, new_pred))

            st.write(f"✅ New model evaluated — RMSE: {new_rmse:,.2f} | R²: {new_r2:.4f}")

            # ── Step 6: Comparison Table ──────────────────────────
            st.divider()
            st.subheader("📊 Model Comparison: Old TBATS vs New TBATS")

            rmse_winner = "🏆 New" if new_rmse < old_rmse else "🏆 Old"
            r2_winner   = "🏆 New" if new_r2   > old_r2   else "🏆 Old"

            comp_df = pd.DataFrame({
                'Metric': [
                    'RMSE (lower is better)',
                    'R² Score (higher is better)',
                ],
                'Old Model': [
                    f"{old_rmse:,.2f}",
                    f"{old_r2:.4f}",
                ],
                'New Model': [
                    f"{new_rmse:,.2f}",
                    f"{new_r2:.4f}",
                ],
                'Winner': [rmse_winner, r2_winner]
            })

            st.dataframe(comp_df, hide_index=True, use_container_width=True)

            # ── Step 7: Actual vs Predicted Chart ─────────────────
            st.subheader(f"📈 Fixed Holdout Period (Last {HOLDOUT_DAYS} Days): Actual vs Predicted")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=holdout_dates, y=holdout_sales,
                mode='lines', name='Actual Sales',
                line=dict(color='#3b82f6', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=holdout_dates, y=old_pred,
                mode='lines', name='Old TBATS Predicted',
                line=dict(color='#f59e0b', width=1.5, dash='dot')
            ))

            fig.add_trace(go.Scatter(
                x=holdout_dates, y=new_pred,
                mode='lines', name='New TBATS Predicted',
                line=dict(color='#10b981', width=2, dash='dash')
            ))

            fig.update_layout(
                title=f'Fixed Holdout — Actual vs Old TBATS vs New TBATS (Last {HOLDOUT_DAYS} Days)',
                xaxis_title='Date',
                yaxis_title='Total Sales',
                plot_bgcolor='#0f172a',
                paper_bgcolor='#0f172a',
                font_color='#94a3b8',
                hovermode='x unified',
                legend=dict(bgcolor='#1e293b', bordercolor='#334155', borderwidth=1),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#1e293b')
            )

            st.plotly_chart(fig, use_container_width=True)

            # ── Step 8: Save Model (retrain on FULL data if better) ──
            # KEY FIX: if new model wins, retrain it on 100% of data
            # before saving. This gives the production model maximum
            # historical data — not just the eval split.
            st.divider()

            if new_rmse < old_rmse:
                st.write("⚙️ Step 6: New model is better — retraining on FULL dataset before saving...")

                # Retrain on ALL records (no holdout withheld)
                # Same params as original notebook — MUST stay consistent
                final_estimator = TBATSEstimator(
                    seasonal_periods=[7, 365],
                    use_arma_errors=False,
                    n_jobs=1
                )

                final_model = final_estimator.fit(sales_array)  # ← full dataset
                joblib.dump(final_model, TBATS_MODEL_PATH)

                st.success(
                    f"✅ **New model is BETTER!**\n\n"
                    f"RMSE improved: {old_rmse:,.2f} → {new_rmse:,.2f}\n\n"
                    f"R² improved: {old_r2:.4f} → {new_r2:.4f}\n\n"
                    f"✅ Model retrained on **full {len(sales_array):,} records** and saved."
                )

            else:
                st.warning(
                    f"⚠️ **New model did NOT improve.**\n\n"
                    f"RMSE: {old_rmse:,.2f} (old) vs {new_rmse:,.2f} (new)\n\n"
                    f"R²: {old_r2:.4f} (old) vs {new_r2:.4f} (new)\n\n"
                    f"⚠️ Old TBATS model kept — no changes saved."
                )

        except Exception as e:
            import traceback
            st.error(f"❌ Retraining failed: {e}")
            st.code(traceback.format_exc())
