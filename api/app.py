from fastapi import FastAPI
import joblib, os, numpy as np

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "/opt/project/models/latest_model.pkl")
_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict(payload: dict):
    model = load_model()
    X = np.array([list(payload["features"].values())])
    proba = float(model.predict_proba(X)[0,1])
    return {"churn_proba": proba, "churn_pred": int(proba >= 0.5)}
