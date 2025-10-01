from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os, mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///opt/mlruns")
DATA_PATH = "/opt/project/data/fleet_churn.csv"
ARTIFACT_DIR = "/opt/project/models"

def step_preprocess(**_):
    from src.data_prep import preprocess
    preprocess(DATA_PATH, "/opt/project/data/processed/train.parquet", "/opt/project/data/processed/val.parquet")

def step_train(**_):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    from src.train import train_model
    return train_model("/opt/project/data/processed/train.parquet",
                       "/opt/project/data/processed/val.parquet",
                       ARTIFACT_DIR)

with DAG(
    dag_id="train_churn_model",
    start_date=datetime(2025, 9, 29),
    schedule_interval="0 2 * * *",
    catchup=False,
    tags=["ml","churn"],
) as dag:
    preprocess = PythonOperator(task_id="preprocess", python_callable=step_preprocess)
    train = PythonOperator(task_id="train", python_callable=step_train)
    preprocess >> train
