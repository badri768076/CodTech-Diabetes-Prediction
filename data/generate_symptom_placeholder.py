"""
SYNTHETIC PLACEHOLDER GENERATOR
================================
This script generates a synthetic dataset that mirrors the SCHEMA of the
"Early Stage Diabetes Risk Prediction Dataset" (Sylhet Diabetes Hospital
survey, available on Kaggle / UCI).

WHY THIS EXISTS:
This sandbox cannot reach kaggle.com (network is restricted to package
registries and GitHub). Rather than skip the symptom-based model, this
script produces statistically plausible placeholder data with the same
columns, so the full pipeline (preprocessing -> training -> API -> UI)
is real and working end-to-end.

TO USE REAL DATA INSTEAD:
1. Download "early_stage_diabetes_risk_prediction_dataset.csv" from Kaggle.
2. Drop it into backend/data/ with that exact filename.
3. Re-run train_symptom_model.py — it will automatically prefer the real
   file over this placeholder (see the loader logic in that script).
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 520  # similar order of magnitude to the real dataset (520 rows)

symptom_cols = [
    "Polyuria", "Polydipsia", "sudden_weight_loss", "weakness", "Polyphagia",
    "Genital_thrush", "visual_blurring", "Itching", "Irritability",
    "delayed_healing", "partial_paresis", "muscle_stiffness", "Alopecia",
    "Obesity",
]

rows = []
for _ in range(n):
    # Base prevalence, then push distribution based on a latent "positive" draw
    is_positive = np.random.rand() < 0.62  # dataset is roughly 61% positive
    age = int(np.clip(np.random.normal(48 if is_positive else 42, 13), 16, 90))
    gender = np.random.choice(["Male", "Female"])
    row = {"Age": age, "Gender": gender}
    for col in symptom_cols:
        base_p = 0.55 if is_positive else 0.25
        row[col] = "Yes" if np.random.rand() < base_p else "No"
    row["class"] = "Positive" if is_positive else "Negative"
    rows.append(row)

df = pd.DataFrame(rows)
out_path = "/home/claude/diabetes-app/backend/data/symptom_data_placeholder.csv"
df.to_csv(out_path, index=False)
print(f"Wrote synthetic placeholder: {out_path} ({len(df)} rows)")
