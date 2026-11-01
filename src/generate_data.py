"""
generate_data.py
-----------------
Creates a realistic (synthetic) 2-year daily dataset for a hostel/institutional
canteen: attendance, menu type, holidays, exams, weather, meals prepared,
meals consumed, and resulting waste/shortage.

In a real deployment, this file would be replaced by an export from the
canteen's attendance/billing system. The generation logic here is written to
mimic real-world patterns (weekday effects, exam-week appetite dips, festival
spikes, monsoon absenteeism, chronic over-preparation) so that the ML models
trained on it behave the way they would on real data.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

START_DATE = datetime(2024, 1, 1)
N_DAYS = 730  # 2 years

BASE_HOSTEL_STRENGTH = 850  # total students registered for mess

# Fixed academic calendar patterns (repeating each year)
HOLIDAY_MONTH_DAYS = {(1, 26), (8, 15), (10, 2), (12, 25), (1, 1), (3, 8), (11, 1)}
EXAM_WINDOWS = [((5, 1), (5, 15)), ((11, 20), (12, 5)), ((2, 1), (2, 10)), ((9, 1), (9, 10))]
FESTIVAL_SPIKES = {(3, 25), (10, 20), (10, 21), (11, 12), (8, 19)}  # special/celebratory menu days
SEMESTER_BREAK = [((6, 1), (6, 20)), ((12, 20), (1, 5))]


def in_window(month_day, windows):
    for (sm, sd), (em, ed) in windows:
        start = (sm, sd)
        end = (em, ed)
        if start <= month_day <= end:
            return True
    return False


def build_dataset():
    rows = []
    for i in range(N_DAYS):
        date = START_DATE + timedelta(days=i)
        month_day = (date.month, date.day)
        dow = date.weekday()  # 0=Mon .. 6=Sun

        is_holiday = 1 if month_day in HOLIDAY_MONTH_DAYS else 0
        is_exam_period = 1 if in_window(month_day, EXAM_WINDOWS) else 0
        is_semester_break = 1 if in_window(month_day, SEMESTER_BREAK) else 0
        is_festival = 1 if month_day in FESTIVAL_SPIKES else 0

        # Weather: crude monsoon simulation (Jun-Sep higher rain probability)
        monsoon = date.month in (6, 7, 8, 9)
        rain_prob = 0.55 if monsoon else 0.15
        is_rainy = np.random.binomial(1, rain_prob)

        # Menu type cycles through the week, with festival override
        menu_cycle = ["North Indian", "South Indian", "North Indian",
                      "Chinese", "South Indian", "Special Thali", "Special Thali"]
        menu_type = "Festival Special" if is_festival else menu_cycle[dow]

        # --- Effective hostel occupancy (students actually on campus) ---
        occupancy = BASE_HOSTEL_STRENGTH
        if is_semester_break:
            occupancy = int(occupancy * np.random.uniform(0.05, 0.15))  # near-empty
        elif is_holiday:
            occupancy = int(occupancy * np.random.uniform(0.55, 0.75))  # many go home
        else:
            occupancy = int(occupancy * np.random.uniform(0.92, 1.0))

        # --- Attendance rate at the mess (fraction of occupants who eat) ---
        base_attend_rate = 0.82
        if dow in (5, 6):  # weekends: more students eat out
            base_attend_rate -= 0.12
        if is_exam_period:
            base_attend_rate += 0.05  # less time to go out, more mess reliance
        if is_rainy:
            base_attend_rate += 0.06  # rain -> fewer people step out to eat
        if menu_type == "Chinese":
            base_attend_rate += 0.03
        if menu_type == "Festival Special":
            base_attend_rate += 0.10
        if is_holiday and not is_semester_break:
            base_attend_rate -= 0.08

        base_attend_rate = np.clip(base_attend_rate + np.random.normal(0, 0.03), 0.35, 0.99)
        attendance = int(occupancy * base_attend_rate)

        # --- True demand (meals actually consumed) ---
        per_person_noise = np.random.normal(1.0, 0.04)
        meals_consumed = max(0, int(attendance * per_person_noise))

        # --- Historical (non-ML) kitchen behaviour: rule-of-thumb over-preparation ---
        # Canteen staff historically prepare based on registered strength with a
        # flat safety margin, not on actual predicted attendance -> this is the
        # inefficiency the ML project targets.
        naive_margin = np.random.uniform(0.12, 0.22)
        meals_prepared = int(meals_consumed * (1 + naive_margin) + np.random.normal(0, 15))
        meals_prepared = max(meals_prepared, int(meals_consumed * 0.85))  # rarely under-shoots badly

        waste = max(0, meals_prepared - meals_consumed)
        shortage = max(0, meals_consumed - meals_prepared)

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "day_of_week": dow,
            "day_name": date.strftime("%A"),
            "is_holiday": is_holiday,
            "is_exam_period": is_exam_period,
            "is_semester_break": is_semester_break,
            "is_festival": is_festival,
            "is_rainy": is_rainy,
            "menu_type": menu_type,
            "hostel_occupancy": occupancy,
            "attendance": attendance,
            "meals_prepared": meals_prepared,
            "meals_consumed": meals_consumed,
            "waste_units": waste,
            "shortage_units": shortage,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "canteen_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head(10).to_string())
    print("\nSummary stats:")
    print(df[["meals_prepared", "meals_consumed", "waste_units", "shortage_units"]].describe())
