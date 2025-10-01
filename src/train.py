import pandas as pd, mlflow, os, joblib
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from pathlib import Path

def train_model(train_pq, val_pq, artifact_dir):
    mlflow.set_experiment("churn_training")
    with mlflow.start_run():
        train = pd.read_parquet(train_pq)
        val = pd.read_parquet(val_pq)
        y_tr, X_tr = train["label"].values, train.drop(columns=["label"])
        y_va, X_va = val["label"].values, val.drop(columns=["label"])
        model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.0, eval_metric="auc", random_state=42, tree_method="hist")
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:,1]
        auc = roc_auc_score(y_va, proba)
        preds = (proba >= 0.5).astype(int)
        f1 = f1_score(y_va, preds)
        mlflow.log_metric("val_auc", float(auc))
        mlflow.log_metric("val_f1", float(f1))
        mlflow.log_params(model.get_params())
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(artifact_dir, "latest_model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")
        return {"model_path": model_path, "val_auc": float(auc), "val_f1": float(f1)}
