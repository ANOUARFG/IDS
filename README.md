# Intrusion Detection System (IDS) Project

## Overview

This repository contains an Intrusion Detection System (IDS) project aimed at identifying and mitigating various types of network attacks by analyzing traffic patterns and detecting anomalies. The IDS provides mechanisms for preprocessing data, training machine learning models, evaluating performance, and visualizing results. This project is suitable for both research and practical implementation in cybersecurity environments.

## Features

- **Data Preprocessing:** Tools to clean, normalize, and prepare datasets for model training.
- **Machine Learning Models:** Implementation of various algorithms (e.g., Random Forest, SVM, Neural Networks) for intrusion detection tasks.
- **Model Evaluation:** Performance metrics, confusion matrix, and ROC curve visualizations.
- **Visualization:** Graphical representation of results, feature importance, and attack distributions.
- **FastAPI Inference API:** Real-time REST API for making predictions on network traffic data.
- **Data Drift Detection:** Automated monitoring and retraining capabilities using Apache Airflow.
- **Modular Design:** Easily extensible for new datasets, models, or evaluation techniques.
- **Documentation:** Well-documented codebase and usage examples.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [FastAPI Inference API](#fastapi-inference-api)
- [Data Drift Detection](#data-drift-detection)
- [Project Structure](#project-structure)
- [Model Training & Evaluation](#model-training--evaluation)
- [Results & Visualization](#results--visualization)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ANOUARFG/IDS.git
   cd IDS
   ```

2. **Set up a Python virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

- The project uses the UNSW-NB15 dataset for training and evaluation.
- Place your dataset in the `data/` directory and update the configuration files or scripts accordingly.

## Usage

### Model Training and Evaluation

1. **Preprocessing:**
   - Run the data preprocessing script to clean and prepare your raw dataset.
   - Example:
     ```bash
     python preprocess.py --input data/raw_dataset.csv --output data/processed_dataset.csv
     ```

2. **Training:**
   - Train your preferred machine learning model using the provided training scripts.
   - Example:
     ```bash
     python train.py --data data/processed_dataset.csv --model random_forest
     ```

3. **Evaluation:**
   - Evaluate the trained model using test data.
   - Example:
     ```bash
     python evaluate.py --model models/random_forest.pkl --test data/test_dataset.csv
     ```

4. **Visualization:**
   - Generate performance plots and feature importance graphs.
   - Example:
     ```bash
     python visualize.py --results results/evaluation.json
     ```

### FastAPI Inference API

The project includes a FastAPI-based REST API for real-time intrusion detection:

1. **Start the API Server:**
   ```bash
   # Using the startup script
   python start_api.py
   
   # Or directly with uvicorn
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the API:**
   - **API Documentation**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/
   - **Model Info**: http://localhost:8000/model/info

3. **Make Predictions:**
   ```python
   import requests
   
   # Sample prediction request
   response = requests.post("http://localhost:8000/predict", json={
       "data": [{"dur": 0.0, "proto": "tcp", "service": "http", ...}]
   })
   result = response.json()
   print(f"Prediction: {result['predictions']}")
   ```

For detailed API documentation, see [src/api/README.md](src/api/README.md).

### Data Drift Detection

The project includes automated data drift detection using Apache Airflow:

1. **Set up Airflow:**
   ```bash
   export AIRFLOW_HOME=./airflow
   airflow db init
   ```

2. **Start Airflow:**
   ```bash
   airflow webserver --port 8080
   airflow scheduler
   ```

3. **Monitor Drift Detection:**
   - Access Airflow UI at http://localhost:8080
   - The `ids_drift_detection` DAG runs hourly to monitor for data drift
   - Automatic model retraining is triggered when drift is detected

## Project Structure

```
IDS/
├── data/                           # Datasets (raw, processed, and test)
│   ├── UNSW-NB15_*.csv            # UNSW-NB15 dataset files
│   ├── processed/                 # Processed data files
│   └── Training and Testing Sets/ # Train/test splits
├── models/                        # Saved trained models and components
│   ├── mlp_model.joblib          # Trained MLP model
│   └── components/               # Model preprocessing components
├── src/                          # Source code
│   ├── api/                      # FastAPI inference API
│   │   ├── app.py               # Main API application
│   │   ├── test_api.py          # API testing script
│   │   └── README.md            # API documentation
│   ├── models/                   # Model-related modules
│   │   └── drift_handler.py     # Data drift detection
│   └── utils/                    # Utility functions
├── dags/                         # Apache Airflow DAGs
│   └── ids_drift_dag.py         # Data drift detection DAG
├── requirements.txt              # Python dependencies
├── start_api.py                  # API startup script
├── Dockerfile                    # Main Docker configuration
├── Dockerfile.api               # API-specific Docker configuration
├── docker-compose.yaml          # Docker Compose configuration
└── README.md                    # This file
```

## Model Training & Evaluation

- **Supported Models:** MLPClassifier (Neural Network), Random Forest, SVM, etc.
- **Hyperparameter Tuning:** Available through scikit-learn GridSearchCV
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Cross-Validation:** K-Fold cross-validation for robust evaluation

## Results & Visualization

- **Performance Metrics:** Comprehensive evaluation with multiple metrics
- **Confusion Matrix:** Visual representation of classification results
- **ROC Curves:** Receiver Operating Characteristic analysis
- **Feature Importance:** Analysis of most predictive features
- **Attack Distribution:** Visualization of attack type distributions

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, new features, or bug fixes.

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

- **Author:** Anouar FARROUG
- **GitHub:** [ANOUARFG](https://github.com/ANOUARFG)
