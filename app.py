"""
app.py
------
A simple Streamlit dashboard so non-technical kitchen staff can get a
recommended quantity to prepare, without touching any code.

Run:
    streamlit run app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "canteen_data.csv")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

st.set_page_config(page_title="Canteen Demand Forecaster", page_icon="🍽️")
st.title("🍽️ Canteen Meal Demand & Waste Reduction")
st.caption("ML-based forecasting to decide how much food to prepare tomorrow.")

@st.cache_resource(show_spinner="Preparing model (first load only)...")
def load_bundle():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        # Pickle saved with different library versions: retrain here
        from train_model import main as train_main
        train_main()
        return joblib.load(MODEL_PATH)


bundle = load_bundle()
model = bundle["model"]
feature_cols = bundle["feature_cols"]
hist = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
recent = hist.tail(7)


@st.cache_data
def get_training_data():
    """Re-runs the same feature engineering as train_model.py to get X_train/y_train,
    so we can retrain a quantile model on demand when the cost sliders change."""
    from train_model import load_and_engineer, get_feature_cols
    df = load_and_engineer()
    cols = get_feature_cols(df)
    split_idx = int(len(df) * 0.8)
    return df[cols].iloc[:split_idx], df["meals_consumed"].iloc[:split_idx]


@st.cache_resource
def get_quantile_model(alpha):
    """Trains a Gradient Boosting quantile regressor for the given target
    quantile. Cached per alpha value so moving the sliders doesn't retrain
    unnecessarily. This is a real ML model learning the recommended quantity
    directly from data -- no statistical formula involved."""
    X_train, y_train = get_training_data()
    qm = GradientBoostingRegressor(loss="quantile", alpha=alpha,
                                    n_estimators=300, max_depth=3,
                                    learning_rate=0.05, random_state=42)
    qm.fit(X_train, y_train)
    return qm


st.sidebar.header("Cost assumptions")
cost_waste = st.sidebar.slider("Cost per wasted meal", 1, 10, 1)
cost_shortage = st.sidebar.slider("Cost per shortage meal", 1, 10, 4)
target_quantile = cost_shortage / (cost_shortage + cost_waste)

st.header("Describe tomorrow")
col1, col2 = st.columns(2)
with col1:
    day_of_week = st.selectbox("Day of week", list(range(7)),
                                format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d])
    menu_type = st.selectbox("Menu type", ["North Indian", "South Indian", "Chinese",
                                            "Special Thali", "Festival Special"])
    hostel_occupancy = st.number_input("Hostel occupancy", min_value=0, max_value=2000, value=800)
with col2:
    is_holiday = st.checkbox("Holiday")
    is_exam_period = st.checkbox("Exam period")
    is_semester_break = st.checkbox("Semester break")
    is_festival = st.checkbox("Festival")
    is_rainy = st.checkbox("Rainy day")

if st.button("Get recommendation", type="primary"):
    row = {
        "day_of_week": day_of_week,
        "is_holiday": int(is_holiday),
        "is_exam_period": int(is_exam_period),
        "is_semester_break": int(is_semester_break),
        "is_festival": int(is_festival),
        "is_rainy": int(is_rainy),
        "hostel_occupancy": hostel_occupancy,
        "month": pd.Timestamp.today().month,
        "day_of_month": pd.Timestamp.today().day,
        "is_weekend": 1 if day_of_week in (5, 6) else 0,
        "consumed_lag1": recent["meals_consumed"].iloc[-1],
        "consumed_lag7": recent["meals_consumed"].iloc[-7] if len(recent) >= 7 else recent["meals_consumed"].iloc[0],
        "consumed_roll7_mean": recent["meals_consumed"].mean(),
        "consumed_roll7_std": recent["meals_consumed"].std(),
        "occupancy_lag1": recent["hostel_occupancy"].iloc[-1],
    }
    for col in feature_cols:
        if col.startswith("menu_"):
            row[col] = 1 if col == f"menu_{menu_type}" else 0

    X = pd.DataFrame([row])[feature_cols].fillna(0)

    predicted_demand = float(model.predict(X)[0])
    quantile_model = get_quantile_model(target_quantile)
    recommended = int(round(quantile_model.predict(X)[0]))

    st.success(f"### Recommended quantity to prepare: **{recommended} meals**")
    st.write(f"Predicted average demand: **{predicted_demand:.0f}** meals")
    st.write(f"Recommended quantity is learned directly by a quantile regression "
         f"model targeting the {target_quantile:.0%} percentile of demand "
         f"(waste to shortage cost ratio {cost_waste} to {cost_shortage}).")
st.divider()
st.subheader("Recent demand trend")
st.line_chart(hist.tail(30).set_index("date")["meals_consumed"])