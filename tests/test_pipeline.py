"""
tests/test_pipeline.py
-----------------------
Basic sanity tests. Run with: pytest
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_data import build_dataset


def test_dataset_shape():
    df = build_dataset()
    assert len(df) == 730
    expected_cols = {"date", "day_of_week", "meals_prepared", "meals_consumed",
                      "waste_units", "shortage_units"}
    assert expected_cols.issubset(set(df.columns))


def test_no_negative_meals():
    df = build_dataset()
    assert (df["meals_consumed"] >= 0).all()
    assert (df["meals_prepared"] >= 0).all()


def test_waste_and_shortage_are_mutually_exclusive_or_zero():
    df = build_dataset()
    # a day can't have both waste and shortage at once
    both = ((df["waste_units"] > 0) & (df["shortage_units"] > 0)).sum()
    assert both == 0
