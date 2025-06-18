from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
import joblib
import h5py
import warnings
import logging
from src.utils.create_test_data import create_test_data
warnings.filterwarnings('ignore')

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.drift_handler import DataDriftHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define default arguments
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
    'ids_drift_detection',
    default_args=default_args,
    description='DAG for IDS data drift detection',
    schedule_interval='@hourly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

def create_test_data_task():
    """Task to create test data with drift"""
    try:
        logger.info("Creating test data...")
        create_test_data()
        logger.info("Test data created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating test data: {str(e)}")
        raise

def load_data():
    """Load new data for drift detection"""
    try:
        data_path = '/opt/airflow/data/processed/new_data.csv'
        logger.info(f"Loading data from: {data_path}")
        
        if not os.path.exists(data_path):
            logger.warning(f"New data not found at {data_path}, falling back to reference data")
            data_path = '/opt/airflow/data/processed/reference_data.csv'
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"No data found at {data_path}")
        
        data = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data with shape: {data.shape}")
        
        # Push the data to XCom
        return data.to_json()
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def detect_drift(**context):
    """Detect drift in the data"""
    try:
        # Get the data from XCom
        data_json = context['task_instance'].xcom_pull(task_ids='load_data')
        if not data_json:
            raise ValueError("No data received from load_data task")
        
        data = pd.read_json(data_json)
        logger.info(f"Processing data with shape: {data.shape}")
        
        # TODO: Implement drift detection logic
        # For now, return a dummy drift score
        drift_score = 0.5
        
        return drift_score
    except Exception as e:
        logger.error(f"Error detecting drift: {str(e)}")
        raise

def check_threshold(**context):
    """Check if drift exceeds threshold"""
    try:
        drift_score = context['task_instance'].xcom_pull(task_ids='detect_drift')
        if drift_score is None:
            raise ValueError("No drift score received from detect_drift task")
        
        threshold = 0.7
        needs_retraining = drift_score > threshold
        
        logger.info(f"Drift score: {drift_score}, Threshold: {threshold}")
        logger.info(f"Model {'needs' if needs_retraining else 'does not need'} retraining")
        
        return needs_retraining
    except Exception as e:
        logger.error(f"Error checking threshold: {str(e)}")
        raise

def retrain_model(**context):
    """Retrain the model if needed"""
    try:
        needs_retraining = context['task_instance'].xcom_pull(task_ids='check_threshold')
        if needs_retraining is None:
            raise ValueError("No retraining decision received from check_threshold task")
        
        if needs_retraining:
            logger.info("Starting model retraining...")
            # TODO: Implement model retraining logic
            logger.info("Model retraining completed")
        else:
            logger.info("No retraining needed")
        
        return True
    except Exception as e:
        logger.error(f"Error in retraining: {str(e)}")
        raise

# Define tasks
create_data_task = PythonOperator(
    task_id='create_test_data',
    python_callable=create_test_data_task,
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

detect_drift_task = PythonOperator(
    task_id='detect_drift',
    python_callable=detect_drift,
    provide_context=True,
    dag=dag,
)

check_threshold_task = PythonOperator(
    task_id='check_threshold',
    python_callable=check_threshold,
    provide_context=True,
    dag=dag,
)

retrain_model_task = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
create_data_task >> load_data_task >> detect_drift_task >> check_threshold_task >> retrain_model_task 