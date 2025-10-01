import pandas as pd, joblib

def batch_predict(input_path, model_path, output_path, threshold=0.5):
    model = joblib.load(model_path)
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)
    proba = model.predict_proba(df)[:,1]
    pred = (proba >= threshold).astype(int)
    out = df.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = pred
    if output_path.endswith(".csv"):
        out.to_csv(output_path, index=False)
    else:
        out.to_parquet(output_path, index=False)
