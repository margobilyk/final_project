from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime

app = FastAPI(title="Police Prediction API")

BASE_DIR = r"C:\Users\maksm\PycharmProjects\PythonProject10\bd"
MODEL_PATH = os.path.join(BASE_DIR, "police_model_auto.pkl")
LOG_FILE = os.path.join(BASE_DIR, "live_logs.csv")

if os.path.exists(MODEL_PATH):
    artifact = joblib.load(MODEL_PATH)
    model = artifact['model']
    threshold = artifact['threshold']
    features = artifact['features']
    print(f"Model loaded from {MODEL_PATH}")
else:
    model = None
    print(f"Error: Model not found at {MODEL_PATH}")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'District', 'hour_sin', 'hour_cos',
                         'day_of_week', 'month_sin', 'month_cos', 'risk_score'])


class PatrolRequest(BaseModel):
    district: int
    hour: int
    day_of_week: int
    month: int


@app.post("/predict")
def predict_risk(req: PatrolRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        input_data = {
            'District': req.district,
            'hour_sin': np.sin(2 * np.pi * req.hour / 24),
            'hour_cos': np.cos(2 * np.pi * req.hour / 24),
            'day_of_week': req.day_of_week,
            'month_sin': np.sin(2 * np.pi * req.month / 12),
            'month_cos': np.cos(2 * np.pi * req.month / 12)
        }

        input_df = pd.DataFrame([input_data])

        prob = model.predict_proba(input_df[features])[0][1]
        alert = prob > threshold

        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(),
                input_data['District'],
                input_data['hour_sin'],
                input_data['hour_cos'],
                input_data['day_of_week'],
                input_data['month_sin'],
                input_data['month_cos'],
                prob
            ])

        return {
            "district": req.district,
            "risk_score": float(round(prob, 4)),
            "alert": bool(alert),
            "action": "DISPATCH" if alert else "MONITOR"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))