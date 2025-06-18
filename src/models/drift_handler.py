import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataDriftDetector:
    def __init__(self):
        """Initialize the drift detector"""
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Define paths relative to project root
        self.reference_data_path = os.path.join(project_root, 'data/processed/reference_data.csv')
        self.metrics_path = os.path.join(project_root, 'data/drift_metrics')
        self.plots_path = os.path.join(project_root, 'data/drift_plots')
        
        # Ensure directories exist
        os.makedirs(self.metrics_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.reference_data_path), exist_ok=True)
        
        # Initialize drift metrics
        self.drift_metrics = {
            'feature_drift': {},
            'target_drift': None,
            'model_performance': None,
            'timestamp': None
        }
    
    def set_reference_data(self, data):
        """Set reference data for drift detection"""
        # Save reference data
        data.to_csv(self.reference_data_path, index=False)
        print(f"Reference data saved to {self.reference_data_path}")
    
    def get_reference_data(self):
        """Get reference data for drift detection"""
        if not os.path.exists(self.reference_data_path):
            raise FileNotFoundError(
                f"Reference data not found at {self.reference_data_path}. "
                "Please set reference data using set_reference_data() method."
            )
        return pd.read_csv(self.reference_data_path)
    
    def detect_feature_drift(self, new_data, reference_data):
        """Detect drift in feature distributions"""
        drift_scores = {}
        
        # Calculate drift for each feature
        for column in reference_data.columns:
            if column != 'label':  # Skip target column
                # Calculate distribution difference
                ref_dist = reference_data[column].value_counts(normalize=True)
                new_dist = new_data[column].value_counts(normalize=True)
                
                # Calculate drift score (KL divergence)
                drift_score = self._calculate_kl_divergence(ref_dist, new_dist)
                drift_scores[column] = drift_score
        
        self.drift_metrics['feature_drift'] = drift_scores
        return drift_scores
    
    def detect_target_drift(self, new_data, reference_data):
        """Detect drift in target distribution"""
        if 'label' in new_data.columns and 'label' in reference_data.columns:
            ref_target_dist = reference_data['label'].value_counts(normalize=True)
            new_target_dist = new_data['label'].value_counts(normalize=True)
            
            drift_score = self._calculate_kl_divergence(ref_target_dist, new_target_dist)
            self.drift_metrics['target_drift'] = drift_score
            return drift_score
        return None
    
    def _calculate_kl_divergence(self, p, q):
        """Calculate KL divergence between two distributions"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Calculate KL divergence
        kl_div = np.sum(p * np.log(p / q))
        return kl_div
    
    def save_drift_metrics(self):
        """Save drift metrics to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.drift_metrics['timestamp'] = timestamp
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'feature': list(self.drift_metrics['feature_drift'].keys()),
            'drift_score': list(self.drift_metrics['feature_drift'].values())
        })
        metrics_df.to_csv(os.path.join(self.metrics_path, f'drift_metrics_{timestamp}.csv'), index=False)
        
        # Save target drift
        with open(os.path.join(self.metrics_path, f'target_drift_{timestamp}.txt'), 'w') as f:
            f.write(f"Target Drift Score: {self.drift_metrics['target_drift']}\n")
            f.write(f"Model Performance: {self.drift_metrics['model_performance']}\n")
    
    def plot_drift_metrics(self):
        """Plot drift metrics"""
        timestamp = self.drift_metrics['timestamp']
        
        # Plot feature drift
        plt.figure(figsize=(12, 6))
        drift_scores = pd.Series(self.drift_metrics['feature_drift'])
        drift_scores.sort_values().plot(kind='bar')
        plt.title('Feature Drift Scores')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, f'feature_drift_{timestamp}.png'))
        plt.close()
        
        # Plot target distribution comparison
        plt.figure(figsize=(10, 6))
        ref_dist = self.get_reference_data()['label'].value_counts(normalize=True)
        ref_dist.plot(kind='bar', label='Reference')
        plt.title('Target Distribution Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, f'target_distribution_{timestamp}.png'))
        plt.close()

class DataDriftHandler:
    def __init__(self):
        """Initialize the drift handler"""
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Define paths relative to project root
        self.model_path = os.path.join(project_root, 'models/mlp_model.joblib')
        self.components_path = os.path.join(project_root, 'models/components')
        self.results_path = os.path.join(project_root, 'models/model_results.h5')
        
        # Initialize components
        self.detector = DataDriftDetector()
        self.model = None
        self.scaler = None
        self.pca = None
        self.selector = None
        self.label_encoders = None
        
        # Load components
        self._load_components()
    
    def _load_components(self):
        """Load model and preprocessing components"""
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(os.path.join(self.components_path, 'scaler.joblib'))
        self.pca = joblib.load(os.path.join(self.components_path, 'pca.joblib'))
        self.selector = joblib.load(os.path.join(self.components_path, 'selector.joblib'))
        self.label_encoders = joblib.load(os.path.join(self.components_path, 'label_encoders.joblib'))
    
    def set_reference_data(self, data):
        """Set reference data for drift detection"""
        self.detector.set_reference_data(data)
    
    def handle_drift(self, new_data, drift_threshold=0.1):
        """Handle data drift by detecting and retraining if necessary"""
        try:
            # Get reference data
            reference_data = self.detector.get_reference_data()
        except FileNotFoundError:
            print("No reference data found. Setting current data as reference...")
            self.set_reference_data(new_data)
            reference_data = new_data
        
        # Detect drift
        feature_drift = self.detector.detect_feature_drift(new_data, reference_data)
        target_drift = self.detector.detect_target_drift(new_data, reference_data)
        
        # Check if drift exceeds threshold
        max_feature_drift = max(feature_drift.values())
        if max_feature_drift > drift_threshold or (target_drift and target_drift > drift_threshold):
            print(f"Drift detected! Max feature drift: {max_feature_drift:.4f}, Target drift: {target_drift:.4f}")
            self._retrain_model(new_data)
        else:
            print("No significant drift detected.")
        
        # Save drift metrics and plots
        self.detector.save_drift_metrics()
        self.detector.plot_drift_metrics()
    
    def _retrain_model(self, new_data):
        """Retrain the model with new data"""
        # Preprocess new data
        X = new_data.drop('label', axis=1)
        y = new_data['label']
        
        # Handle categorical features
        for column in X.select_dtypes(include=['object']).columns:
            X[column] = self.label_encoders[column].transform(X[column].astype(str))
        
        # Apply feature selection
        X_selected = self.selector.transform(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        
        # Apply PCA
        X_pca = self.pca.transform(X_scaled)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
        
        # Retrain model
        self.model.fit(X_train, y_train)
        
        # Evaluate new model
        cv_scores = cross_val_score(self.model, X_pca, y, cv=5)
        test_score = self.model.score(X_test, y_test)
        
        # Update model results
        self.detector.drift_metrics['model_performance'] = {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'test_score': test_score
        }
        
        # Save updated model
        joblib.dump(self.model, self.model_path)
        print(f"Model retrained. New test score: {test_score:.4f}")

def main():
    # Example usage
    handler = DataDriftHandler()
    
    # Load new data
    new_data = pd.read_csv('data/processed/new_data.csv')
    
    # Handle drift (will set reference data if it doesn't exist)
    handler.handle_drift(new_data)

if __name__ == "__main__":
    main() 