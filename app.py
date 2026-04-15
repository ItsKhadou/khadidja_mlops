import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("artifacts/model.pkl")

app = FastAPI(
    title="Mushroom Prediction API",
    description="Mini projet MLOps — prédiction edible vs poisonous",
    version="1.0.0",
)

_artifact = None


def load_model():
    global _artifact
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. Lance d'abord train.py."
        )
    with open(MODEL_PATH, "rb") as f:
        _artifact = pickle.load(f)


@app.on_event("startup")
def startup_event():
    load_model()


class PredictRequest(BaseModel):
    features: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_loaded": _artifact is not None}


@app.post("/predict")
def predict(body: PredictRequest):
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    model = _artifact["model"]
    feature_names = _artifact["feature_names"]

    missing_features = [col for col in feature_names if col not in body.features]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Colonnes manquantes : {missing_features}"
        )

    input_df = pd.DataFrame([[body.features[col] for col in feature_names]], columns=feature_names)

    pred = model.predict(input_df)[0]

    return {
        "prediction": pred
    }