FROM apache/airflow:2.9.3
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.12.txt"

# Let constraints choose compatible versions; do NOT pin pyarrow or fsspec/gcsfs here.
# Leave mlflow unpinned so it matches constraints/pyarrow in the base image.
RUN pip install --no-cache-dir --constraint ${CONSTRAINT_URL} \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    pandas==2.1.4 \
    scikit-learn==1.5.1 \
    mlflow \
    fastapi==0.112.2 \
    uvicorn==0.30.6 \
    joblib==1.4.2 \
    jupyterlab==4.2.4