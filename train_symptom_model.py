"""
Trains the SYMPTOM-BASED model (Age, Gender + 14 Yes/No symptom fields)
matching the schema of the "Early Stage Diabetes Risk Prediction Dataset"
(Sylhet Diabetes Hospital survey, Kaggle/UCI).

DATA SOURCE PRIORITY:
1. data/early_stage_diabetes_risk_prediction_dataset.csv  <- real Kaggle file,
   used automatically if you place it here (see data/generate_symptom_placeholder.py
   for why this isn't bundled by default).
2. data/symptom_data_placeholder.csv                       <- synthetic fallback,
   lets the pipeline run end-to-end without the real file.

Run: python3 train_symptom_model.py
Outputs: model/symptom_model.joblib, model/symptom_encoders.joblib,
         model/symptom_metrics.json
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

REAL_PATH = "data/early_stage_diabetes_risk_prediction_dataset.csv"
PLACEHOLDER_PATH = "data/symptom_data_placeholder.csv"

SYMPTOM_COLS = [
    "Polyuria", "Polydipsia", "sudden_weight_loss", "weakness", "Polyphagia",
    "Genital_thrush", "visual_blurring", "Itching", "Irritability",
    "delayed_healing", "partial_paresis", "muscle_stiffness", "Alopecia",
    "Obesity",
]
ALL_FEATURES = ["Age", "Gender"] + SYMPTOM_COLS

def load_data():
    if os.path.exists(REAL_PATH):
        print(f"Using REAL Kaggle dataset: {REAL_PATH}")
        df = pd.read_csv(REAL_PATH)
        using_real = True
    else:
        print(f"Real Kaggle file not found at {REAL_PATH}")
        print(f"Falling back to SYNTHETIC PLACEHOLDER: {PLACEHOLDER_PATH}")
        df = pd.read_csv(PLACEHOLDER_PATH)
        using_real = False
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df, using_real

def main():
    df, using_real = load_data()

    label_col = "class" if "class" in df.columns else df.columns[-1]
    encoders = {}
    for col in ["Gender"] + SYMPTOM_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

    y_le = LabelEncoder()
    y = y_le.fit_transform(df[label_col].astype(str).str.strip())
    positive_label = "Positive" if "Positive" in y_le.classes_ else y_le.classes_[-1]
    positive_idx = list(y_le.classes_).index(positive_label)
    if positive_idx == 0:
        y = 1 - y  # ensure 1 == positive/diabetic

    X = df[ALL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=2,
        class_weight="balanced", random_state=42
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "data_source": "real_kaggle_csv" if using_real else "SYNTHETIC_PLACEHOLDER",
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": len(X_train), "n_test": len(X_test),
    }
    print(classification_report(y_test, y_pred, target_names=["Non-diabetic", "Diabetic"]))
    print(json.dumps(metrics, indent=2))

    joblib.dump(model, "model/symptom_model.joblib")
    joblib.dump(encoders, "model/symptom_encoders.joblib")
    with open("model/symptom_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("model/symptom_features.json", "w") as f:
        json.dump(ALL_FEATURES, f)

    if not using_real:
        print("\n*** NOTE: this model was trained on SYNTHETIC placeholder data. ***")
        print("*** Replace with the real Kaggle CSV and re-run for production use. ***")

if __name__ == "__main__":
    main()
