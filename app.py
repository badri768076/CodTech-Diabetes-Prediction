"""
Diabetes Risk Prediction API
=============================
Serves two models trained on different datasets/schemas and a combined
endpoint that ensembles them when both lab values and symptoms are given.

Run:  uvicorn app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

app = FastAPI(title="Diabetes Risk Prediction API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- Load artifacts ----------
lab_model = joblib.load(MODEL_DIR / "lab_model.joblib")
lab_imputer = joblib.load(MODEL_DIR / "lab_imputer.joblib")
lab_scaler = joblib.load(MODEL_DIR / "lab_scaler.joblib")
lab_features = json.load(open(MODEL_DIR / "lab_features.json"))
lab_metrics = json.load(open(MODEL_DIR / "lab_metrics.json"))

symptom_model = joblib.load(MODEL_DIR / "symptom_model.joblib")
symptom_encoders = joblib.load(MODEL_DIR / "symptom_encoders.joblib")
symptom_features = json.load(open(MODEL_DIR / "symptom_features.json"))
symptom_metrics = json.load(open(MODEL_DIR / "symptom_metrics.json"))

SYMPTOM_COLS = [c for c in symptom_features if c not in ("Age", "Gender")]


# ---------- Schemas ----------
class LabInput(BaseModel):
    Pregnancies: float = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=0, le=300)
    BloodPressure: float = Field(..., ge=0, le=200)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=900)
    BMI: float = Field(..., ge=0, le=70)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3)
    Age: float = Field(..., ge=1, le=120)


class SymptomInput(BaseModel):
    Age: float = Field(..., ge=1, le=120)
    Gender: str
    Polyuria: str
    Polydipsia: str
    sudden_weight_loss: str
    weakness: str
    Polyphagia: str
    Genital_thrush: str
    visual_blurring: str
    Itching: str
    Irritability: str
    delayed_healing: str
    partial_paresis: str
    muscle_stiffness: str
    Alopecia: str
    Obesity: str


class CombinedInput(BaseModel):
    lab: Optional[LabInput] = None
    symptoms: Optional[SymptomInput] = None


# ---------- Helpers ----------
def predict_lab(payload: LabInput) -> float:
    import pandas as pd  # noqa: local import keeps top-level deps minimal
    x = pd.DataFrame([[getattr(payload, f) for f in lab_features]], columns=lab_features)
    x_imp = lab_imputer.transform(x)
    x_scaled = lab_scaler.transform(x_imp)
    return float(lab_model.predict_proba(x_scaled)[0, 1])


def predict_symptoms(payload: SymptomInput) -> float:
    row = []
    for f in symptom_features:
        val = getattr(payload, f)
        if f == "Age":
            row.append(float(val))
        else:
            enc_map = symptom_encoders[f]
            key = str(val).strip()
            if key not in enc_map:
                # case-insensitive fallback (Yes/No, Male/Female)
                match = next((k for k in enc_map if k.lower() == key.lower()), None)
                if match is None:
                    raise HTTPException(400, f"Invalid value '{val}' for field '{f}'. Allowed: {list(enc_map.keys())}")
                key = match
            row.append(enc_map[key])
    x = np.array([row])
    return float(symptom_model.predict_proba(x)[0, 1])


def risk_band(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.6:
        return "Moderate"
    return "High"


# ---------- Routes ----------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info():
    return {
        "lab_model": {
            "features": lab_features,
            "selected_algorithm": lab_metrics.get("selected_model"),
            "test_roc_auc": lab_metrics.get("test_roc_auc"),
            "test_recall": lab_metrics.get("test_recall"),
            "trained_on": "Pima Indians Diabetes Database (real data)",
        },
        "symptom_model": {
            "features": symptom_features,
            "test_roc_auc": symptom_metrics.get("test_roc_auc"),
            "test_recall": symptom_metrics.get("test_recall"),
            "trained_on": symptom_metrics.get("data_source"),
        },
    }


@app.post("/api/predict/lab")
def predict_lab_endpoint(payload: LabInput):
    prob = predict_lab(payload)
    return {"probability": round(prob, 4), "risk_band": risk_band(prob), "source": "lab_model"}


@app.post("/api/predict/symptoms")
def predict_symptoms_endpoint(payload: SymptomInput):
    prob = predict_symptoms(payload)
    return {"probability": round(prob, 4), "risk_band": risk_band(prob), "source": "symptom_model"}


@app.post("/api/predict/combined")
def predict_combined(payload: CombinedInput):
    if payload.lab is None and payload.symptoms is None:
        raise HTTPException(400, "Provide at least one of 'lab' or 'symptoms'.")

    results = {}
    if payload.lab is not None:
        results["lab_probability"] = round(predict_lab(payload.lab), 4)
    if payload.symptoms is not None:
        results["symptom_probability"] = round(predict_symptoms(payload.symptoms), 4)

    if "lab_probability" in results and "symptom_probability" in results:
        # Weighted ensemble — lab model gets more weight, it has the stronger,
        # real-data-validated ROC-AUC of the two.
        combined = 0.6 * results["lab_probability"] + 0.4 * results["symptom_probability"]
    elif "lab_probability" in results:
        combined = results["lab_probability"]
    else:
        combined = results["symptom_probability"]

    results["combined_probability"] = round(combined, 4)
    results["risk_band"] = risk_band(combined)
    return results


# ---------- Serve frontend ----------
FRONTEND_DIR = BASE_DIR.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
