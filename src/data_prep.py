import pandas as pd
from pathlib import Path

def preprocess(raw_csv, train_out, val_out, val_ratio=0.2, seed=42):
    df = pd.read_csv(raw_csv)
    df = df.dropna(subset=["churn"])
    y = df["churn"].astype(int)
    X = df.drop(columns=["churn","customer_id"], errors="ignore")
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    X[cat_cols] = X[cat_cols].fillna("NA")
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)
    n = len(X)
    val_n = max(1, int(n * val_ratio))
    X_train, X_val = X.iloc[:-val_n], X.iloc[-val_n:]
    y_train, y_val = y.iloc[:-val_n], y.iloc[-val_n:]
    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True).rename("label")], axis=1).to_parquet(train_out, index=False)
    pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True).rename("label")], axis=1).to_parquet(val_out, index=False)
