import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define required directories
directories = [
    os.path.join(project_root, 'data/processed'),
    os.path.join(project_root, 'data/drift_metrics'),
    os.path.join(project_root, 'data/drift_plots')
]

# Create directories
print("Creating project directory structure...")
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Define paths
processed_data_path = os.path.join(project_root, 'data/processed')
reference_data_path = os.path.join(processed_data_path, 'reference_data.csv')
components_path = os.path.join(project_root, 'models/components')

def load_components():
    """Load preprocessing components"""
    try:
        # Load components
        scaler = joblib.load(os.path.join(components_path, 'scaler.joblib'))
        pca = joblib.load(os.path.join(components_path, 'pca.joblib'))
        selector = joblib.load(os.path.join(components_path, 'selector.joblib'))
        label_encoders = joblib.load(os.path.join(components_path, 'label_encoders.joblib'))
        
        print("Successfully loaded preprocessing components")
        return scaler, pca, selector, label_encoders
    except FileNotFoundError as e:
        print(f"\nError loading components: {str(e)}")
        print("\nPlease ensure you have run the data processing in Google Colab and")
        print("downloaded the components to:", components_path)
        return None, None, None, None

def setup_drift_detection():
    """Setup drift detection by creating reference data if needed"""
    try:
        # Load components
        scaler, pca, selector, label_encoders = load_components()
        if scaler is None:
            return False
        
        # Check if we have the model results file
        model_results_path = os.path.join(components_path, 'model_results.h5')
        if not os.path.exists(model_results_path):
            print("\nModel results not found. Please ensure you have:")
            print("1. Run the model training in Google Colab")
            print("2. Downloaded the model results file")
            print("3. Placed it in:", components_path)
            return False
        
        # Check if we have the processed data
        processed_data_file = os.path.join(processed_data_path, 'processed_data.csv')
        if not os.path.exists(processed_data_file):
            print("\nProcessed data not found. Please ensure you have:")
            print("1. Run the data processing in Google Colab")
            print("2. Downloaded the processed data files")
            print("3. Placed them in:", processed_data_path)
            return False
        
        # Load processed data
        data = pd.read_csv(processed_data_file)
        print(f"Successfully loaded processed data with shape: {data.shape}")
        
        # Create reference data if it doesn't exist
        if not os.path.exists(reference_data_path):
            print("\nCreating reference data from first batch...")
            reference_data = data.head(10000)  # Use first 10000 samples as reference
            reference_data.to_csv(reference_data_path, index=False)
            print(f"Reference data saved to {reference_data_path}")
        
        return True
    
    except Exception as e:
        print(f"\nError during drift detection setup: {str(e)}")
        return False

def main():
    print("Setting up drift detection...")
    if setup_drift_detection():
        print("\nDrift detection setup completed successfully!")
        print("\nRequired files and directories:")
        print(f"1. Components directory: {components_path}")
        print(f"2. Processed data: {processed_data_path}")
        print(f"3. Reference data: {reference_data_path}")
        print(f"4. Drift metrics: {os.path.join(project_root, 'data/drift_metrics')}")
        print(f"5. Drift plots: {os.path.join(project_root, 'data/drift_plots')}")
    else:
        print("\nDrift detection setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 