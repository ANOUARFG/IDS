from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import List, Dict, Any, Optional
import uvicorn
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IDS Inference API",
    description="Intrusion Detection System API for real-time network traffic analysis",
    version="1.0.0"
)

# Instrument FastAPI with Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]
    
class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]
    confidence_scores: List[float]
    timestamp: str
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_type: str
    features: List[str]
    classes: List[str]
    accuracy: Optional[float]
    last_trained: Optional[str]

# Global variables for model and components
model = None
scaler = None
pca = None
selector = None
label_encoders = None
feature_names = None
class_names = None

def load_model_components():
    """Load all model components"""
    global model, scaler, pca, selector, label_encoders, feature_names, class_names
    
    try:
        # Load model components
        model_path = "models/mlp_model.joblib"
        scaler_path = "models/components/scaler.joblib"
        pca_path = "models/components/pca.joblib"
        selector_path = "models/components/selector.joblib"
        encoders_path = "models/components/label_encoders.joblib"
        
        if not all(os.path.exists(path) for path in [model_path, scaler_path, pca_path, selector_path, encoders_path]):
            logger.error("One or more model components not found")
            return False
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        selector = joblib.load(selector_path)
        label_encoders = joblib.load(encoders_path)
        
        # Load feature names (you might need to adjust this based on your data)
        feature_names = [
            'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 
            'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 
            'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 
            'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 
            'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 
            'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 
            'ct_srv_dst', 'is_sm_ips_ports'
        ]
        
        # Define class names based on UNSW-NB15 dataset
        class_names = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms']
        
        logger.info("All model components loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        return False

def preprocess_data(data: List[Dict[str, Any]]) -> np.ndarray:
    """Preprocess input data for prediction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Encode categorical variables
        categorical_columns = ['proto', 'service', 'state']
        for col in categorical_columns:
            if col in df.columns and col in label_encoders:
                df[col] = label_encoders[col].transform(df[col].astype(str))
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only the required features in the correct order
        df = df[feature_names]
        
        # Scale the features
        df_scaled = scaler.transform(df)
        
        # Apply feature selection
        df_selected = selector.transform(df_scaled)
        
        # Apply PCA
        df_pca = pca.transform(df_selected)
        
        return df_pca
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Data preprocessing error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model components on startup"""
    logger.info("Starting IDS Inference API...")
    if not load_model_components():
        logger.error("Failed to load model components")
        raise RuntimeError("Model components could not be loaded")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information"""
    return ModelInfoResponse(
        model_type="MLPClassifier",
        features=feature_names,
        classes=class_names,
        accuracy=None,  # You can add this if you have it stored
        last_trained=None  # You can add this if you have it stored
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions on network traffic data"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess the data
        processed_data = preprocess_data(request.data)
        
        # Make predictions
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)
        
        # Calculate confidence scores (max probability for each prediction)
        confidence_scores = np.max(probabilities, axis=1).tolist()
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist(),
            confidence_scores=confidence_scores,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make batch predictions (with background processing)"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # For now, just return the same as single prediction
        # You can implement background processing here if needed
        return await predict(request)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/features")
async def get_features():
    """Get list of required features"""
    return {
        "features": feature_names,
        "categorical_features": ['proto', 'service', 'state'],
        "numerical_features": [f for f in feature_names if f not in ['proto', 'service', 'state']]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 