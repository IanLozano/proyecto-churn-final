from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

# Cargar pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "pipeline_model.pkl")
MODEL_PATH = os.path.abspath(MODEL_PATH)

print("Ruta del modelo:", MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Pydantic
class Record(BaseModel):
    Contract: str
    Tenure: float
    MonthlyCharges: float

class BatchIn(BaseModel):
    records: List[Record]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

def run_model(df: pd.DataFrame):
    # Alinear nombres de columnas con el pipeline
    df = df.copy()
    # El pipeline se entrenó con 'tenure' en minúsculas
    df.rename(columns={"Tenure": "tenure"}, inplace=True)

    # Asegurar tipos
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["Contract"] = df["Contract"].astype(str).str.strip()

    proba = model.predict_proba(df)[:, 1]
    return proba.tolist()

@app.post("/predict_batch")
def predict_batch(req: BatchIn):
    if not req.records:
        raise HTTPException(status_code=400, detail="no records")

    df = pd.DataFrame([r.model_dump() for r in req.records])
    try:
        probas = run_model(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error al predecir: {e}")

    return {"probas": probas}

@app.post("/predict")
def predict_one(rec: Record):
    df = pd.DataFrame([rec.model_dump()])
    try:
        proba = run_model(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error al predecir: {e}")
    return {"proba": proba}
