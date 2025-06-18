import requests
import json
import pandas as pd
from typing import List, Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_model_info():
    """Test the model info endpoint"""
    print("Testing model info...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_features():
    """Test the features endpoint"""
    print("Testing features endpoint...")
    response = requests.get(f"{BASE_URL}/features")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def create_sample_data() -> List[Dict[str, Any]]:
    """Create sample network traffic data for testing"""
    return [
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
        },
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

def test_prediction():
    """Test the prediction endpoint"""
    print("Testing prediction endpoint...")
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Make prediction request
    payload = {"data": sample_data}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predictions: {result['predictions']}")
        print(f"Confidence Scores: {result['confidence_scores']}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Model Version: {result['model_version']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_batch_prediction():
    """Test the batch prediction endpoint"""
    print("Testing batch prediction endpoint...")
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Make batch prediction request
    payload = {"data": sample_data}
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Batch Predictions: {result['predictions']}")
        print(f"Batch Confidence Scores: {result['confidence_scores']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def main():
    """Run all tests"""
    print("Starting API Tests...")
    print("=" * 50)
    
    try:
        test_health_check()
        test_model_info()
        test_features()
        test_prediction()
        test_batch_prediction()
        
        print("All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    main() 