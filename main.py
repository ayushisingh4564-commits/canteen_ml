"""
main.py
-------
One-command entry point: generates data (skip if you already have real data
in data/canteen_data.csv), trains all models, evaluates them, and saves
outputs. This is the file you (or a grader/recruiter) runs first.

Usage:
    python3 main.py              # generate synthetic data + train
    python3 main.py --no-generate  # use existing data/canteen_data.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data import build_dataset
from train_model import main as train_main


def run(generate=True):
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    if generate:
        print(">>> Generating synthetic dataset...")
        df = build_dataset()
        df.to_csv("data/canteen_data.csv", index=False)
        print(f">>> Saved {len(df)} rows to data/canteen_data.csv\n")
    else:
        if not os.path.exists("data/canteen_data.csv"):
            raise FileNotFoundError(
                "data/canteen_data.csv not found. Run without --no-generate, "
                "or place your real dataset there first."
            )
        print(">>> Using existing data/canteen_data.csv\n")

    print(">>> Training models...")
    train_main()
    print("\n>>> Done. See outputs/ for the trained model, metrics.json, and plots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-generate", action="store_true",
                         help="Skip synthetic data generation; use existing data/canteen_data.csv")
    args = parser.parse_args()
    run(generate=not args.no_generate)
