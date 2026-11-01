"""
train_model.py
---------------
Loads the canteen dataset, engineers features, trains several demand
forecasting models, and picks the best one. Also trains a SEPARATE quantile
regression model (Gradient Boosting with quantile loss) that directly learns
to predict the recommended preparation quantity -- i.e. HOW MUCH to cook, not
just the expected average demand. This keeps the entire pipeline machine
learning end-to-end: no classical statistics formula (e.g. assuming a normal
distribution) is used anywhere -- the quantile model *learns* the shape of
the demand distribution from data instead.

Outputs:
  outputs/model.pkl                  -> best point-forecast model + quantile model
  outputs/plots/feature_importance.png
  outputs/plots/actual_vs_predicted.png
  outputs/plots/waste_comparison.png
  outputs/metrics.json
  outputs/recommendations_sample.csv
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_PATH = os.path.join(BASE_DIR, "data", "canteen_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)

# Business assumption: wasting a meal costs less than a shortage (an unhappy /
# unfed student, reputational cost). Tune these to your canteen's real costs.
COST_PER_WASTED_MEAL = 1.0
COST_PER_SHORTAGE_MEAL = 4.0
# The quantile a "recommended quantity" model should target, derived from the
# cost ratio: if shortage is 4x worse than waste, we want to satisfy demand
# 4/(4+1) = 80% of the time -> train a model that predicts the 80th percentile
# of demand directly, instead of assuming a distribution shape mathematically.
TARGET_QUANTILE = COST_PER_SHORTAGE_MEAL / (COST_PER_SHORTAGE_MEAL + COST_PER_WASTED_MEAL)


def load_and_engineer():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Lag / rolling features an ML pipeline would have available the night
    # before (i.e., no leakage from same-day attendance/meals_consumed).
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["consumed_lag1"] = df["meals_consumed"].shift(1)
    df["consumed_lag7"] = df["meals_consumed"].shift(7)
    df["consumed_roll7_mean"] = df["meals_consumed"].shift(1).rolling(7).mean()
    df["consumed_roll7_std"] = df["meals_consumed"].shift(1).rolling(7).std()
    df["occupancy_lag1"] = df["hostel_occupancy"].shift(1)

    df = pd.get_dummies(df, columns=["menu_type"], prefix="menu")

    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_cols(df):
    exclude = {"date", "day_name", "meals_prepared", "meals_consumed",
               "waste_units", "shortage_units", "attendance"}
    return [c for c in df.columns if c not in exclude]


def train_and_evaluate(df, feature_cols):
    X = df[feature_cols]
    y = df["meals_consumed"]

    # Time-based split: train on first ~80% of days, test on the most recent ~20%
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=8,
                                               random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                        learning_rate=0.05, random_state=42),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}
        fitted[name] = model

    # Naive baseline: "same as last week" for comparison
    naive_preds = X_test["consumed_lag7"]
    results["NaiveLastWeek"] = {
        "MAE": mean_absolute_error(y_test, naive_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, naive_preds)),
        "R2": r2_score(y_test, naive_preds),
        "MAPE_%": np.mean(np.abs((y_test - naive_preds) / y_test)) * 100,
    }

    best_name = min(
        (n for n in results if n != "NaiveLastWeek"),
        key=lambda n: results[n]["MAE"]
    )
    return results, fitted, best_name, (X_train, X_test, y_train, y_test)


def train_quantile_model(X_train, y_train, quantile):
    """
    Trains a Gradient Boosting model with quantile loss. Unlike the point-
    estimate models above (which predict the mean/expected demand), this
    model is trained to directly predict a target percentile of demand --
    e.g. a value that actual demand exceeds only ~20% of the time (for
    quantile=0.8). This is itself a machine learning model (gradient boosted
    trees), just trained with a different loss function -- there is no
    statistical formula or distribution assumption involved.
    """
    model = GradientBoostingRegressor(
        loss="quantile", alpha=quantile,
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def main():
    df = load_and_engineer()
    feature_cols = get_feature_cols(df)
    results, fitted, best_name, splits = train_and_evaluate(df, feature_cols)
    X_train, X_test, y_train, y_test = splits
    best_model = fitted[best_name]

    print("=== Model comparison (test set, most recent ~20% of days) ===")
    for name, m in results.items():
        print(f"{name:18s} MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  "
              f"R2={m['R2']:.3f}  MAPE={m['MAPE_%']:.1f}%")
    print(f"\nBest model: {best_name}")

    # Residual std kept only for reporting/diagnostics (not used to compute
    # the recommendation any more -- that now comes from the quantile model).
    test_preds = best_model.predict(X_test)
    residuals = y_test.values - test_preds
    residual_std = float(np.std(residuals))

    # --- Train the quantile regression model: this directly LEARNS the
    # recommended preparation quantity from data (pure ML, no formula). ---
    quantile_model = train_quantile_model(X_train, y_train, TARGET_QUANTILE)
    recommended = quantile_model.predict(X_test)

    # --- Simulate cost outcomes: historical rule-of-thumb vs ML recommendation ---
    test_df = df.iloc[X_test.index].copy()
    test_df["predicted_demand"] = test_preds
    test_df["recommended_prepare"] = np.round(recommended).astype(int)

    hist_waste = np.maximum(0, test_df["meals_prepared"] - test_df["meals_consumed"])
    hist_shortage = np.maximum(0, test_df["meals_consumed"] - test_df["meals_prepared"])
    hist_cost = hist_waste * COST_PER_WASTED_MEAL + hist_shortage * COST_PER_SHORTAGE_MEAL

    ml_waste = np.maximum(0, test_df["recommended_prepare"] - test_df["meals_consumed"])
    ml_shortage = np.maximum(0, test_df["meals_consumed"] - test_df["recommended_prepare"])
    ml_cost = ml_waste * COST_PER_WASTED_MEAL + ml_shortage * COST_PER_SHORTAGE_MEAL

    summary = {
        "best_model": best_name,
        "model_metrics": results,
        "residual_std": residual_std,
        "quantile_model": "GradientBoostingRegressor(loss='quantile')",
        "target_quantile": float(TARGET_QUANTILE),
        "historical_avg_daily_waste": float(hist_waste.mean()),
        "historical_avg_daily_shortage": float(hist_shortage.mean()),
        "historical_avg_daily_cost": float(hist_cost.mean()),
        "ml_avg_daily_waste": float(ml_waste.mean()),
        "ml_avg_daily_shortage": float(ml_shortage.mean()),
        "ml_avg_daily_cost": float(ml_cost.mean()),
        "waste_reduction_%": float((1 - ml_waste.mean() / hist_waste.mean()) * 100),
        "cost_reduction_%": float((1 - ml_cost.mean() / hist_cost.mean()) * 100),
    }

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    joblib.dump({
        "model": best_model,
        "quantile_model": quantile_model,
        "feature_cols": feature_cols,
        "residual_std": residual_std,
        "target_quantile": TARGET_QUANTILE,
    }, f"{OUT_DIR}/model.pkl")

    test_df[["date", "day_name", "meals_consumed", "meals_prepared",
             "predicted_demand", "recommended_prepare"]].to_csv(
        f"{OUT_DIR}/recommendations_sample.csv", index=False)

    # --- Plots ---
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(12)
        plt.figure(figsize=(8, 6))
        fi.plot(kind="barh", color="#4C72B0")
        plt.title(f"Top feature importances ({best_name})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/plots/feature_importance.png", dpi=140)
        plt.close()
    elif hasattr(best_model, "coef_"):
        coef = pd.Series(best_model.coef_, index=feature_cols)
        top = coef.reindex(coef.abs().sort_values(ascending=True).tail(12).index)
        plt.figure(figsize=(8, 6))
        colors = ["#C44E52" if v < 0 else "#4C72B0" for v in top.values]
        top.plot(kind="barh", color=colors)
        plt.title(f"Top feature coefficients ({best_name})")
        plt.xlabel("Coefficient (effect on predicted meals)")
        plt.axvline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/plots/feature_importance.png", dpi=140)
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(test_df["date"].values, y_test.values, label="Actual demand", linewidth=1.8)
    plt.plot(test_df["date"].values, test_preds, label="Predicted demand", linewidth=1.5, linestyle="--")
    plt.title("Actual vs Predicted Meal Demand (test period)")
    plt.xlabel("Date")
    plt.ylabel("Meals consumed")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/actual_vs_predicted.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6, 5))
    bars = plt.bar(
        ["Historical\n(rule of thumb)", "ML-recommended"],
        [summary["historical_avg_daily_waste"], summary["ml_avg_daily_waste"]],
        color=["#C44E52", "#55A868"]
    )
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                  f"{b.get_height():.0f}", ha="center")
    plt.title("Average Daily Food Waste: Before vs After ML")
    plt.ylabel("Meals wasted / day")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/waste_comparison.png", dpi=140)
    plt.close()

    print("\n=== Waste / cost simulation on held-out test period ===")
    print(f"Historical avg daily waste:    {summary['historical_avg_daily_waste']:.1f} meals")
    print(f"ML avg daily waste:            {summary['ml_avg_daily_waste']:.1f} meals")
    print(f"Waste reduction:               {summary['waste_reduction_%']:.1f}%")
    print(f"Historical avg daily shortage: {summary['historical_avg_daily_shortage']:.2f} meals")
    print(f"ML avg daily shortage:         {summary['ml_avg_daily_shortage']:.2f} meals")
    print(f"Cost reduction:                {summary['cost_reduction_%']:.1f}%")


if __name__ == "__main__":
    main()
