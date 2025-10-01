from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

def step_inference(**_):
    from src.predict import batch_predict
    batch_predict("/opt/project/data/scoring/input.csv",
                  "/opt/project/models/latest_model.pkl",
                  "/opt/project/data/scoring/predictions.csv")

with DAG(
    dag_id="churn_inference_pipeline",
    start_date=datetime(2025, 9, 29),
    schedule_interval="0 * * * *",
    catchup=False,
    tags=["ml","inference"],
) as dag:
    predict = PythonOperator(task_id="batch_predict", python_callable=step_inference)
