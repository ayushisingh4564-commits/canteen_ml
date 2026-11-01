"""
predict_next_day.py
--------------------
Simulates how this system is used operationally: the previous night, kitchen
staff (or an automated job) enters/knows tomorrow's known context (day of
week, menu, holiday/exam flags, expected weather, hostel occupancy) and gets
back a recommended quantity to prepare.

Run: python3 predict_next_day.py
Edit TOMORROW dict below to try different scenarios.
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "canteen_data.csv")

# ----- Edit this block to describe tomorrow -----
TOMORROW = {
    "day_of_week": 5,          # 0=Mon ... 5=Sat
    "is_holiday": 0,
    "is_exam_period": 0,
    "is_semester_break": 0,
    "is_festival": 1,          # e.g., a festival special menu day
    "is_rainy": 0,
    "menu_type": "Festival Special",
    "hostel_occupancy": 820,
    "month": 11,
    "day_of_month": 12,
}
# --------------------------------------------------


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    quantile_model = bundle["quantile_model"]
    feature_cols = bundle["feature_cols"]

    hist = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    recent = hist.tail(7)

    row = {
        "day_of_week": TOMORROW["day_of_week"],
        "is_holiday": TOMORROW["is_holiday"],
        "is_exam_period": TOMORROW["is_exam_period"],
        "is_semester_break": TOMORROW["is_semester_break"],
        "is_festival": TOMORROW["is_festival"],
        "is_rainy": TOMORROW["is_rainy"],
        "hostel_occupancy": TOMORROW["hostel_occupancy"],
        "month": TOMORROW["month"],
        "day_of_month": TOMORROW["day_of_month"],
        "is_weekend": 1 if TOMORROW["day_of_week"] in (5, 6) else 0,
        "consumed_lag1": recent["meals_consumed"].iloc[-1],
        "consumed_lag7": recent["meals_consumed"].iloc[-7] if len(recent) >= 7 else recent["meals_consumed"].iloc[0],
        "consumed_roll7_mean": recent["meals_consumed"].mean(),
        "consumed_roll7_std": recent["meals_consumed"].std(),
        "occupancy_lag1": recent["hostel_occupancy"].iloc[-1],
    }

    for col in feature_cols:
        if col.startswith("menu_"):
            row[col] = 1 if col == f"menu_{TOMORROW['menu_type']}" else 0

    X = pd.DataFrame([row])[feature_cols].fillna(0)

    # Two ML models, two different jobs:
    predicted_demand = float(model.predict(X)[0])              # expected/average demand
    recommended = int(round(quantile_model.predict(X)[0]))       # learned recommended quantity

    print("=== Tomorrow's context ===")
    for k, v in TOMORROW.items():
        print(f"  {k}: {v}")
    print("\n=== Recommendation ===")
    print(f"Predicted demand (expected meals consumed): {predicted_demand:.0f}")
    print(f"Recommended quantity (learned by quantile regression model): {recommended}")
    print(f"==> RECOMMENDED QUANTITY TO PREPARE: {recommended} meals")


if __name__ == "__main__":
    main()
