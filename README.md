# Churn MLOps (Airflow + Custom Image, v2)

This version builds a custom Airflow image with your ML stack installed using official Airflow constraints to avoid dependency conflicts.

## Quickstart
1. Put dataset at `data/fleet_churn.csv` (rename your `fleet_churn_30.csv` accordingly).
2. Build the image and start:
   ```bash
   docker compose build
   docker compose up -d postgres redis
   docker compose up -d airflow-webserver airflow-scheduler airflow-triggerer
   ```
3. Initialize and create admin:
   ```bash
   docker compose exec airflow-webserver airflow db init
   docker compose exec airflow-webserver airflow users create      --username admin --firstname Olivia --lastname Chen      --role Admin --email you@example.com --password admin
   ```
4. Open Airflow UI: http://localhost:8080
5. (Optional) JupyterLab: `docker compose up -d jupyter` → http://localhost:8888

## Pipelines
- `train_churn_model`: preprocess → train (XGBoost) → MLflow metrics → save `models/latest_model.pkl`
- `churn_inference_pipeline`: score `data/scoring/input.csv` → write `data/scoring/predictions.csv`
