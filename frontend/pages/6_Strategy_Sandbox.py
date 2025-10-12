import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go
import sys
import traceback
import math

# Add project root to path so src is importable
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.churnxai.preprocess import ChurnDataPreprocessor

st.set_page_config(page_title="Strategy Sandbox", layout="wide")

# --- Load model + preprocessor only (cached) ---
@st.cache_resource(ttl=3600)
def load_artifacts():
    try:
        model = joblib.load(project_root / "artifacts/models/champion_model_xgb.joblib")
        preprocessor = joblib.load(project_root / "artifacts/models/preprocessor.joblib")
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"FATAL ERROR: Could not load artifact: {e}")
        return None, None

model, preprocessor = load_artifacts()

st.title("💡 Strategy Sandbox: A Live 'What-If' Engine")
st.markdown("Design a retention campaign and simulate its real-time impact on churn risk.")

# --- Helper: find matching row indices (first pass) ---
def find_matching_indices(file_path: Path, filters: dict, max_rows: int = 10000, chunksize: int = 200_000):
    """
    First pass: read only the filter columns to cheaply find up to max_rows global indices
    that match the filters. Returns a list of global row indices (0-based, excluding header).
    """
    filter_cols = ['age_in_years', 'days_tenure', 'income']
    matching_indices = []
    row_offset = 0
    total_rows_scanned = 0

    try:
        # Optional: compute total rows for progress display
        try:
            with open(file_path, 'rb') as f:
                total_lines = sum(1 for _ in f)
            total_rows = max(0, total_lines - 1)
        except Exception:
            total_rows = None

        progress_bar = st.progress(0)
        status = st.empty()

        reader = pd.read_csv(file_path, usecols=filter_cols, chunksize=chunksize)
        for chunk in reader:
            # compute boolean mask for chunk
            mask = (
                chunk['age_in_years'].between(filters['age_min'], filters['age_max']) &
                chunk['days_tenure'].between(filters['tenure_min'], filters['tenure_max']) &
                chunk['income'].between(filters['income_min'], filters['income_max'])
            )
            if mask.any():
                # global indices for rows in this chunk
                matched_local_idx = chunk.index[mask]
                for local_idx in matched_local_idx:
                    matching_indices.append(row_offset + int(local_idx))
                    if len(matching_indices) >= max_rows:
                        break
            row_offset += len(chunk)
            total_rows_scanned += len(chunk)

            # update progress and status
            if total_rows:
                progress_bar.progress(min(1.0, total_rows_scanned / total_rows))
                status.text(f"Scanned {total_rows_scanned:,} / {total_rows:,} rows — collected {len(matching_indices):,} matches")
            else:
                # If we couldn't compute total_rows, show scanned count
                progress_bar.progress(0.5)  # indeterminate visual
                status.text(f"Scanned {total_rows_scanned:,} rows — collected {len(matching_indices):,} matches")

            if len(matching_indices) >= max_rows:
                break

        progress_bar.empty()
        status.empty()
        return matching_indices

    except Exception:
        traceback.print_exc()
        st.error("Error while scanning filter columns. See console for traceback.")
        return []

# --- Helper: collect full rows for selected indices (second pass) ---
def collect_rows_by_indices(file_path: Path, indices_set: set, chunksize: int = 100_000):
    """
    Second pass: iterate CSV in chunks and collect rows whose global index is in indices_set.
    Returns a DataFrame with the collected full rows in original order.
    """
    if not indices_set:
        return pd.DataFrame()

    collected = []
    row_offset = 0
    remaining = set(indices_set)  # mutable copy

    try:
        reader = pd.read_csv(file_path, chunksize=chunksize)
        for chunk in reader:
            # local indices in chunk
            local_indices = chunk.index
            # compute global indices for this chunk
            globals_for_chunk = [row_offset + int(i) for i in local_indices]
            # find which ones intersect with remaining
            # build a boolean mask
            mask = [g in remaining for g in globals_for_chunk]
            if any(mask):
                matched = chunk.loc[mask].copy()
                # determine original global positions for ordering (row_offset + local index)
                matched['_global_idx'] = [row_offset + int(i) for i in matched.index]
                collected.append(matched)
                # remove found indices
                for g in matched['_global_idx'].tolist():
                    if g in remaining:
                        remaining.remove(g)
            row_offset += len(chunk)
            if not remaining:
                break
        if collected:
            df = pd.concat(collected, ignore_index=True)
            df = df.sort_values('_global_idx').drop(columns=['_global_idx'])
            return df.reset_index(drop=True)
        else:
            return pd.DataFrame()
    except Exception:
        traceback.print_exc()
        st.error("Error while collecting full rows. See console for traceback.")
        return pd.DataFrame()

# --- Main UI: interactive filters and action ---
if model is None or preprocessor is None:
    st.error("Could not load model/preprocessor artifacts. Sandbox disabled.")
else:
    st.sidebar.header("1. Target Segment Filters")
    age_range = st.sidebar.slider("Target Age Range", 18, 100, (18, 40))
    tenure_range_days = st.sidebar.slider("Target Tenure (Days)", 0, 10000, (0, 730))
    income_range = st.sidebar.slider("Target Annual Income ($)", 0, 200000, (0, 50000), step=1000)

    st.sidebar.header("2. Retention Action")
    premium_discount_pct = st.sidebar.slider("Offer Premium Discount (%)", 0, 50, 10)

    st.sidebar.header("3. Performance / Safety")
    max_rows = st.sidebar.number_input("Max customers to simulate (lower = safer)", min_value=1000, max_value=50000, value=10000, step=1000)
    first_pass_chunksize = st.sidebar.selectbox("First-pass chunksize (rows)", [50_000, 100_000, 200_000], index=1)
    second_pass_chunksize = st.sidebar.selectbox("Second-pass chunksize (rows)", [50_000, 100_000, 200_000], index=1)

    # Prepare filter dict for streaming
    filters = {
        'age_min': age_range[0], 'age_max': age_range[1],
        'tenure_min': tenure_range_days[0], 'tenure_max': tenure_range_days[1],
        'income_min': income_range[0], 'income_max': income_range[1]
    }

    if st.button("🚀 Run Live Simulation (stream-safe)", use_container_width=True):
        file_path = project_root / "data/raw/autoinsurance_churn.csv"
        if not file_path.exists():
            st.error("Raw data file not found. Please place autoinsurance_churn.csv in data/raw/")
        else:
            with st.spinner("Scanning dataset for target customers (fast pass)..."):
                matching_indices = find_matching_indices(file_path, filters, max_rows=int(max_rows), chunksize=int(first_pass_chunksize))
            if not matching_indices:
                st.warning("No matching customers found for these filters.")
            else:
                st.info(f"Found {len(matching_indices):,} matching rows. Collecting full records...")
                with st.spinner("Collecting full records for matched customers..."):
                    df_segment = collect_rows_by_indices(file_path, set(matching_indices), chunksize=int(second_pass_chunksize))

                if df_segment.empty:
                    st.error("Could not collect full rows for matched indices.")
                else:
                    # downcast heavy numeric columns to reduce memory
                    numeric_cols = ['curr_ann_amt', 'income', 'days_tenure', 'age_in_years']
                    for c in numeric_cols:
                        if c in df_segment.columns:
                            df_segment[c] = pd.to_numeric(df_segment[c], errors='coerce').astype('float32')

                    try:
                        # BEFORE
                        X_original = preprocessor.transform(df_segment)
                        try:
                            X_original = X_original[model.get_booster().feature_names]
                        except Exception:
                            pass
                        risk_original = model.predict_proba(X_original)[:, 1]

                        # AFTER: apply discount in-memory then re-transform/predict
                        df_modified = df_segment.copy()
                        if 'curr_ann_amt' in df_modified.columns:
                            df_modified['curr_ann_amt'] = df_modified['curr_ann_amt'] * (1 - premium_discount_pct / 100)

                        X_modified = preprocessor.transform(df_modified)
                        try:
                            X_modified = X_modified[model.get_booster().feature_names]
                        except Exception:
                            pass
                        risk_new = model.predict_proba(X_modified)[:, 1]

                        # KPIs
                        customers_saved = int(((risk_original > 0.5) & (risk_new <= 0.5)).sum())
                        cost_of_campaign = float((df_segment['curr_ann_amt'] * (premium_discount_pct / 100)).sum())

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Target Segment Size", f"{len(df_segment):,} Customers")
                        col2.metric("Potential Customers Saved", f"{customers_saved:,}")
                        col3.metric("Estimated Campaign Cost", f"${cost_of_campaign:,.0f}")

                        st.markdown("#### Churn Risk Distribution: Before vs. After")
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=risk_original, name='Original Risk', marker_color='#EF553B', opacity=0.6))
                        fig.add_trace(go.Histogram(x=risk_new, name='New Risk (After Discount)', marker_color='#00CC96', opacity=0.6))
                        fig.update_layout(barmode='overlay', xaxis_title='Churn Probability', yaxis_title='Number of Customers')
                        st.plotly_chart(fig, use_container_width=True)

                        st.success("Simulation complete.")
                    except MemoryError:
                        st.error("MemoryError: segment still too large for this machine. Try lowering 'Max customers to simulate' and rerun.")
                    except Exception as e:
                        st.error(f"An error occurred during simulation: {e}")
                        traceback.print_exc()
