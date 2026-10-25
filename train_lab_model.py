"""
Trains the LAB-VALUES model (Pregnancies, Glucose, BloodPressure,
SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) on the
real Pima Indians Diabetes Database.

Run: python3 train_lab_model.py
Outputs: model/lab_model.joblib, model/lab_imputer.joblib,
         model/lab_scaler.joblib, model/lab_metrics.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

DATA_PATH = "data/pima.csv"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
# Columns where 0 is not physiologically valid -> treat as missing
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)
    return df

def main():
    df = load_and_clean()
    X = df[FEATURES]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_model, best_auc = None, None, -1
    cv_summary = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring="roc_auc")
        cv_summary[name] = {"mean_auc": float(scores.mean()), "std_auc": float(scores.std())}
        if scores.mean() > best_auc:
            best_auc, best_name, best_model = scores.mean(), name, model

    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "selected_model": best_name,
        "cv_results": cv_summary,
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

    joblib.dump(best_model, "model/lab_model.joblib")
    joblib.dump(imputer, "model/lab_imputer.joblib")
    joblib.dump(scaler, "model/lab_scaler.joblib")
    with open("model/lab_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("model/lab_features.json", "w") as f:
        json.dump(FEATURES, f)

    print(f"\nSelected model: {best_name} (CV ROC-AUC={best_auc:.3f})")
    print("Saved model/lab_model.joblib, lab_imputer.joblib, lab_scaler.joblib")

if __name__ == "__main__":
    main()
