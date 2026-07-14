"""
Data cleaning and feature engineering for the Startup Success Prediction project.
Mirrors the logic developed interactively in notebooks/startup_success_prediction.ipynb.
"""

import numpy as np
import pandas as pd

DROP_COLS = [
    "Unnamed: 0", "id", "object_id",
    "Unnamed: 6",
    "state_code.1",
    "zip_code", "latitude", "longitude",
    "name", "city",
    "closed_at",
    "status",
]

DATE_COLS = ["founded_at", "first_funding_at", "last_funding_at"]
AGE_COLS = [
    "age_first_funding_year", "age_last_funding_year",
    "age_first_milestone_year", "age_last_milestone_year",
]
ROUND_FLAGS = ["has_VC", "has_angel", "has_roundA", "has_roundB", "has_roundC", "has_roundD"]


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage/identifier columns, parse dates, impute missing milestone ages."""
    df = df.drop(columns=DROP_COLS)

    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c])

    df["age_first_milestone_year"] = df["age_first_milestone_year"].fillna(0)
    df["age_last_milestone_year"] = df["age_last_milestone_year"].fillna(0)

    for c in AGE_COLS:
        df[c] = df[c].clip(lower=0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and encode remaining categoricals."""
    df = df.copy()

    df["funding_duration_days"] = (
        (df["last_funding_at"] - df["first_funding_at"]).dt.days.clip(lower=0)
    )
    df["funding_per_relationship"] = df["funding_total_usd"] / (df["relationships"] + 1)
    df["milestone_velocity"] = df["milestones"] / (df["age_last_milestone_year"] + 1)
    df["total_round_types"] = df[ROUND_FLAGS].sum(axis=1)
    df["funding_total_usd_log"] = np.log1p(df["funding_total_usd"])

    cat_freq = df["category_code"].value_counts(normalize=True)
    df["category_code_freq"] = df["category_code"].map(cat_freq)

    state_freq = df["state_code"].value_counts(normalize=True)
    df["state_code_freq"] = df["state_code"].map(state_freq)

    df = df.drop(columns=["founded_at", "first_funding_at", "last_funding_at",
                           "category_code", "state_code"])
    return df


def run_pipeline(raw_csv_path: str) -> pd.DataFrame:
    df = load_raw(raw_csv_path)
    df = clean(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    processed = run_pipeline("data/startup_data.csv")
    processed.to_csv("data/startup_data_processed.csv", index=False)
    print(f"Processed dataset shape: {processed.shape}")
    print("Saved to data/startup_data_processed.csv")
