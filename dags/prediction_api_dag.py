from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import logging
import json
import os

# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'ids_prediction_api',
    default_args=default_args,
    description='DAG to call FastAPI IDS prediction endpoint',
    schedule_interval=None,  # Set to None for manual trigger, or use a cron schedule
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

API_URL = "http://host.docker.internal:8000/predict"  # Use host.docker.internal if Airflow is in Docker

# Sample data for prediction (should match your model's expected features)
sample_data = [
    {
        "dur": 0.0,
        "proto": "tcp",
        "service": "http",
        "state": "FIN",
        "spkts": 2,
        "dpkts": 0,
        "sbytes": 0,
        "dbytes": 0,
        "rate": 0.0,
        "sttl": 254,
        "dttl": 0,
        "sload": 0.0,
        "dload": 0.0,
        "sloss": 0,
        "dloss": 0,
        "sinpkt": 0.0,
        "dinpkt": 0.0,
        "sjit": 0.0,
        "djit": 0.0,
        "swin": 0,
        "stcpb": 0,
        "dtcpb": 0,
        "dwin": 0,
        "tcprtt": 0.0,
        "synack": 0.0,
        "ackdat": 0.0,
        "smean": 0,
        "dmean": 0,
        "trans_depth": 0,
        "response_body_len": 0,
        "ct_srv_src": 2,
        "ct_state_ttl": 0,
        "ct_dst_ltm": 1,
        "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 1,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 0,
        "ct_src_ltm": 1,
        "ct_srv_dst": 2,
        "is_sm_ips_ports": 0
    }
]

def call_prediction_api(**context):
    """Call the FastAPI prediction endpoint and log the result"""
    try:
        payload = {"data": sample_data}
        logging.info(f"Sending payload to API: {json.dumps(payload)[:200]}...")
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        logging.info(f"Prediction result: {json.dumps(result, indent=2)}")
        # Optionally push to XCom
        context['ti'].xcom_push(key='prediction_result', value=result)
        return result
    except Exception as e:
        logging.error(f"Error calling prediction API: {str(e)}")
        raise

def save_result_to_file(**context):
    result = context['ti'].xcom_pull(task_ids='call_prediction_api', key='prediction_result')
    output_dir = '/opt/airflow/data/processed'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'prediction_result.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    logging.info(f"Prediction result saved to {output_path}")

call_api_task = PythonOperator(
    task_id='call_prediction_api',
    python_callable=call_prediction_api,
    provide_context=True,
    dag=dag,
)

save_result_task = PythonOperator(
    task_id='save_result_to_file',
    python_callable=save_result_to_file,
    provide_context=True,
    dag=dag,
)

call_api_task >> save_result_task 