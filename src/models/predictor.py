import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class IDSPredictor:
    def __init__(self):
        """Initialize the IDS predictor"""
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Define paths relative to project root
        self.model_path = os.path.join(project_root, 'models/mlp_model.joblib')
        self.components_path = os.path.join(project_root, 'models/components')
        self.results_path = os.path.join(project_root, 'models/model_results.h5')
        
        # Ensure directories exist
        os.makedirs(self.components_path, exist_ok=True)
        
        # Load model and components
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(os.path.join(self.components_path, 'scaler.joblib'))
        self.pca = joblib.load(os.path.join(self.components_path, 'pca.joblib'))
        self.selector = joblib.load(os.path.join(self.components_path, 'selector.joblib'))
        self.label_encoders = joblib.load(os.path.join(self.components_path, 'label_encoders.joblib'))
    
    def preprocess_input(self, data):
        """Preprocess input data for prediction"""
        # Handle categorical features
        for column in data.select_dtypes(include=['object']).columns:
            if column != 'label':  # Skip target column if present
                data[column] = self.label_encoders[column].transform(data[column].astype(str))
        
        # Apply feature selection
        X_selected = self.selector.transform(data)
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        
        # Apply PCA
        X_pca = self.pca.transform(X_scaled)
        
        return X_pca
    
    def predict(self, data):
        """Make predictions on input data"""
        # Preprocess input
        X_processed = self.preprocess_input(data)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def get_model_info(self):
        """Get information about the model and its components"""
        return {
            'model_type': type(self.model).__name__,
            'n_features': self.selector.n_features_in_,
            'n_components': self.pca.n_components_,
            'feature_names': self.selector.feature_names_in_.tolist()
        }

def main():
    # Example usage
    predictor = IDSPredictor()
    
    # Print model information
    print("\nModel Information:")
    print(f"Model Type: {predictor.get_model_info()['model_type']}")
    print(f"Number of Features: {predictor.get_model_info()['n_features']}")
    print(f"Number of PCA Components: {predictor.get_model_info()['n_components']}")
    print(f"Feature Names: {predictor.get_model_info()['feature_names']}")
    
    # Example of how to use the predictor
    print("\nTo use the predictor with new data:")
    print("1. Load your data into a pandas DataFrame")
    print("2. Make sure it has the same features as the training data")
    print("3. Use the predict method:")
    print("   predictions = predictor.predict(your_data)")
    print("4. The predictions will be the attack categories")

if __name__ == "__main__":
    main() 