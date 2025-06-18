import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json
import os

class DataDriftDetector:
    def __init__(self, window_size=1000, drift_threshold=0.05, save_path='drift_metrics'):
        """
        Initialize the data drift detector
        
        Parameters:
        -----------
        window_size : int
            Size of the sliding window for drift detection
        drift_threshold : float
            Threshold for considering drift significant
        save_path : str
            Path to save drift metrics and plots
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_stats = None
        self.drift_history = []
        self.save_path = save_path
        self.timestamps = []
        
        # Create directory for saving metrics if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=os.path.join(save_path, 'drift_detection.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def compute_statistics(self, data):
        """Compute basic statistics for drift detection"""
        stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'cov': np.cov(data.T)
        }
        return stats
    
    def detect_drift(self, current_data, timestamp=None):
        """
        Detect drift in the current data window compared to reference data
        
        Parameters:
        -----------
        current_data : numpy.ndarray
            Current data window to check for drift
        timestamp : datetime, optional
            Timestamp for the current window
            
        Returns:
        --------
        dict
            Dictionary containing drift metrics and detection result
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if self.reference_stats is None:
            self.reference_stats = self.compute_statistics(current_data)
            self.timestamps.append(timestamp)
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'timestamp': timestamp
            }
        
        current_stats = self.compute_statistics(current_data)
        
        # Compute drift metrics
        mean_drift = np.mean(np.abs(current_stats['mean'] - self.reference_stats['mean']))
        std_drift = np.mean(np.abs(current_stats['std'] - self.reference_stats['std']))
        
        # Compute covariance drift using Mahalanobis distance
        try:
            robust_cov = MinCovDet().fit(current_data)
            mahalanobis_dist = robust_cov.mahalanobis(current_data)
            cov_drift = np.mean(mahalanobis_dist)
        except:
            cov_drift = 0
        
        # Combine drift metrics
        total_drift = (mean_drift + std_drift + cov_drift) / 3
        
        # Store drift history
        self.drift_history.append(total_drift)
        self.timestamps.append(timestamp)
        
        # Check if drift exceeds threshold
        drift_detected = total_drift > self.drift_threshold
        
        if drift_detected:
            logging.info(f"Drift detected! Drift score: {total_drift:.4f}")
            # Update reference statistics
            self.reference_stats = current_stats
        
        return {
            'drift_detected': drift_detected,
            'drift_score': float(total_drift),
            'timestamp': timestamp.isoformat(),
            'metrics': {
                'mean_drift': float(mean_drift),
                'std_drift': float(std_drift),
                'cov_drift': float(cov_drift)
            }
        }
    
    def save_metrics(self):
        """Save drift metrics to a JSON file"""
        metrics = {
            'drift_history': self.drift_history,
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'threshold': self.drift_threshold
        }
        
        with open(os.path.join(self.save_path, 'drift_metrics.json'), 'w') as f:
            json.dump(metrics, f)
    
    def plot_drift_history(self, save_plot=True):
        """Plot the history of drift detection"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.drift_history)
        plt.axhline(y=self.drift_threshold, color='r', linestyle='--', label='Drift Threshold')
        plt.title('Data Drift History')
        plt.xlabel('Time')
        plt.ylabel('Drift Score')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.save_path, 'drift_history.png'))
        plt.close()

class DataDriftHandler:
    def __init__(self, window_size=1000, drift_threshold=0.05, save_path='drift_metrics'):
        """
        Initialize the data drift handler
        
        Parameters:
        -----------
        window_size : int
            Size of the sliding window for drift detection
        drift_threshold : float
            Threshold for considering drift significant
        save_path : str
            Path to save drift metrics and plots
        """
        self.detector = DataDriftDetector(window_size, drift_threshold, save_path)
        self.scaler = RobustScaler()
    
    def process_window(self, data, timestamp=None):
        """
        Process a window of data for drift detection and handling
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data window to process
        timestamp : datetime, optional
            Timestamp for the current window
            
        Returns:
        --------
        tuple
            (processed_data, drift_metrics)
        """
        # Detect drift
        drift_metrics = self.detector.detect_drift(data, timestamp)
        
        if drift_metrics['drift_detected']:
            # Apply drift adaptation
            processed_data = self.scaler.fit_transform(data)
        else:
            processed_data = data
        
        return processed_data, drift_metrics
    
    def process_batch(self, X, timestamps=None):
        """
        Process a batch of data for drift detection and handling
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        timestamps : list of datetime, optional
            List of timestamps for each window
            
        Returns:
        --------
        tuple
            (processed_data, drift_metrics_list)
        """
        n_samples = len(X)
        processed_data = []
        drift_metrics_list = []
        
        for i in range(0, n_samples, self.detector.window_size):
            end_idx = min(i + self.detector.window_size, n_samples)
            current_window = X[i:end_idx]
            
            timestamp = timestamps[i] if timestamps is not None else None
            processed_window, metrics = self.process_window(current_window, timestamp)
            
            processed_data.append(processed_window)
            drift_metrics_list.append(metrics)
        
        # Save metrics and plot
        self.detector.save_metrics()
        self.detector.plot_drift_history()
        
        return np.vstack(processed_data), drift_metrics_list 