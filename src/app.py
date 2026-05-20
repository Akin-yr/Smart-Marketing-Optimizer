import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import time
import os
sys.path.append('src')
from forecast_pipeline import predict_model

st.set_page_config(page_title="Smart Marketing Optimizer", layout="wide")
st.title("Smart Marketing Revenue Forecast Dashboard")
st.write("AI-powered revenue forecasting and marketing campaign optimization system")

@st.cache_resource
def load_bundle():
    return joblib.load('src/model_bundle.pkl')

bundle = load_bundle()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

db_input = st.sidebar.text_input(
    "🔗 PostgreSQL URL (leave blank to use default):",
    type="password"
)

# Priority: UI input > env var > streamlit secrets > local fallback
def resolve_db_url(ui_input: str) -> str:
    if ui_input:
        return ui_input
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    if os.environ.get("SUPABASE_URL"):
        return os.environ["SUPABASE_URL"]
    try:
        if "DATABASE_URL" in st.secrets:
            return st.secrets["DATABASE_URL"]
    except Exception:
        pass
    # Fallback local / Docker Compose default
    return "postgresql://postgres:123456@localhost:5432/smart_marketing_db"

db_url = resolve_db_url(db_input)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Forecast Date Range")

default_start = pd.Timestamp("2023-01-01").date()
default_end   = pd.Timestamp("2024-07-01").date()

start_date = st.sidebar.date_input("From:", value=default_start)
end_date   = st.sidebar.date_input("To:", value=default_end)

if start_date >= end_date:
    st.sidebar.error("⚠️ Start date must be before end date.")
    st.stop()

# ── Forecast Button ───────────────────────────────────────────
if st.button("🚀 Run Forecast", type="primary"):
    with st.spinner("AI model is processing..."):
        try:
            t0 = time.time()
            result = predict_model(
                start_date=pd.Timestamp(start_date),
                end_date=pd.Timestamp(end_date),
                db_url=db_url,
                bundle=bundle,
            )
            elapsed = time.time() - t0

            st.success(f"✅ Forecast completed ({elapsed:.1f}s) — {len(result)} days")

            total_rev  = result["revenue"].sum()
            total_cogs = result["cogs"].sum()
            avg_margin = (total_rev - total_cogs) / total_rev * 100 if total_rev > 0 else 0

            k1, k2, k3 = st.columns(3)
            k1.metric("💰 Total Revenue", f"{total_rev:,.0f}")
            k2.metric("📦 Total COGS",    f"{total_cogs:,.0f}")
            k3.metric("📊 Gross Margin",  f"{avg_margin:.1f}%")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📋 Forecast Table")
                st.dataframe(result, use_container_width=True)
                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv,
                    file_name=f"forecast_{start_date}_{end_date}.csv",
                    mime="text/csv",
                )

            with col2:
                st.subheader("📈 Revenue & COGS Chart")
                st.line_chart(
                    data=result.set_index("date")[["revenue", "cogs"]],
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
