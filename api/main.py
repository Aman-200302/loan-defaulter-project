from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import uvicorn
import os

# ── Load model and feature columns ───────────────────────────────
MODEL_PATH    = os.getenv("MODEL_PATH", "models/model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_columns.pkl")

model           = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts probability of loan default for an applicant",
    version="1.0.0"
)

# ── Request schema ────────────────────────────────────────────────
class ApplicantFeatures(BaseModel):
    person_age:                  float = Field(..., example=28.0)
    person_income:               float = Field(..., example=55000.0)
    person_emp_length:           float = Field(..., example=3.0)
    loan_amnt:                   float = Field(..., example=10000.0)
    loan_int_rate:               float = Field(..., example=12.5)
    loan_percent_income:         float = Field(..., example=0.18)
    cb_person_cred_hist_length:  float = Field(..., example=4.0)
    person_home_ownership:       float = Field(..., example=1.0)
    loan_intent:                 float = Field(..., example=2.0)
    loan_grade:                  float = Field(..., example=1.0)
    cb_person_default_on_file:   float = Field(..., example=0.0)

# ── Response schema ───────────────────────────────────────────────
class PredictionResponse(BaseModel):
    applicant_id:      int
    default_probability: float
    prediction:        str
    risk_level:        str

# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Loan Default Prediction API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantFeatures, applicant_id: int = 0):
    try:
        # build input DataFrame in correct column order
        input_data = pd.DataFrame(
            [[getattr(applicant, col) for col in feature_columns]],
            columns=feature_columns
        )

        proba      = model.predict_proba(input_data)[0][1]
        prediction = "DEFAULT" if proba >= 0.5 else "NO DEFAULT"

        # risk bucketing — useful for business teams
        if proba < 0.2:
            risk = "LOW"
        elif proba < 0.5:
            risk = "MEDIUM"
        elif proba < 0.75:
            risk = "HIGH"
        else:
            risk = "VERY HIGH"

        return PredictionResponse(
            applicant_id=applicant_id,
            default_probability=round(float(proba), 4),
            prediction=prediction,
            risk_level=risk
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(applicants: list[ApplicantFeatures]):
    results = []
    for i, applicant in enumerate(applicants):
        input_data = pd.DataFrame(
            [[getattr(applicant, col) for col in feature_columns]],
            columns=feature_columns
        )
        proba      = model.predict_proba(input_data)[0][1]
        prediction = "DEFAULT" if proba >= 0.5 else "NO DEFAULT"
        results.append({
            "applicant_id": i,
            "default_probability": round(float(proba), 4),
            "prediction": prediction
        })
    return {"predictions": results, "total": len(results)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)