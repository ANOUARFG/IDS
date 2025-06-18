#!/usr/bin/env python3
"""
Startup script for the IDS Inference API
"""

import uvicorn
import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Start the IDS Inference API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Log level")
    
    args = parser.parse_args()
    
    # Check if model files exist
    model_files = [
        "models/mlp_model.joblib",
        "models/components/scaler.joblib",
        "models/components/pca.joblib",
        "models/components/selector.joblib",
        "models/components/label_encoders.joblib"
    ]
    
    missing_files = [f for f in model_files if not Path(f).exists()]
    if missing_files:
        print("Error: Missing model files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all model files are present before starting the API.")
        sys.exit(1)
    
    print("Starting IDS Inference API...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"Log Level: {args.log_level}")
    print("-" * 50)
    
    # Start the server
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True
    )

if __name__ == "__main__":
    main() 