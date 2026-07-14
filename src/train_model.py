"""
Trains and evaluates multiple classifiers for the Startup Success Prediction project,
then saves the best-performing model (by F1 score) along with its preprocessing artifacts.

Usage:
    python src/train_model.py
"""

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocessing import run_pipeline


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  random_state=42, eval_metric="logloss"),
    }


def main():
    print("Loading and processing data...")
    df = run_pipeline("data/startup_data.csv")

    X = df.drop(columns=["labels"])
    y = df["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    print("Training models...\n")
    models = get_models()
    results = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        fitted[name] = model

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        })

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_columns.pkl")

    raw_df = pd.read_csv("data/startup_data.csv")
    cat_freq = raw_df["category_code"].value_counts(normalize=True).to_dict()
    state_freq = raw_df["state_code"].value_counts(normalize=True).to_dict()
    joblib.dump(cat_freq, "models/category_freq_map.pkl")
    joblib.dump(state_freq, "models/state_freq_map.pkl")

    print("\nSaved model + preprocessing artifacts to models/")


if __name__ == "__main__":
    main()
