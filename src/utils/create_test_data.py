import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def get_data_paths():
    """Get the correct paths for data files in Airflow environment"""
    # Check if we're running in Airflow
    if os.path.exists('/opt/airflow'):
        base_path = '/opt/airflow'
    else:
        # Get project root directory for local development
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define paths
    processed_dir = os.path.join(base_path, 'data/processed')
    reference_path = os.path.join(processed_dir, 'reference_data.csv')
    new_data_path = os.path.join(processed_dir, 'new_data.csv')
    
    # Create directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    return reference_path, new_data_path

def create_reference_data():
    """Create reference data with normal distribution"""
    reference_path, _ = get_data_paths()
    
    print("Creating reference data...")
    
    # Create synthetic data
    n_samples = 10000
    n_features = 20
    
    # Generate numerical features
    data = {}
    for i in range(n_features):
        if i < 5:  # First 5 features are network traffic metrics
            data[f'traffic_{i}'] = np.random.normal(100, 20, n_samples)
        elif i < 10:  # Next 5 features are packet statistics
            data[f'packet_{i-5}'] = np.random.gamma(2, 2, n_samples)
        elif i < 15:  # Next 5 features are connection features
            data[f'conn_{i-10}'] = np.random.exponential(5, n_samples)
        else:  # Last 5 features are protocol features
            data[f'proto_{i-15}'] = np.random.binomial(1, 0.3, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add categorical features
    protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS']
    df['protocol'] = np.random.choice(protocols, n_samples)
    
    services = ['web', 'mail', 'dns', 'ftp', 'ssh']
    df['service'] = np.random.choice(services, n_samples)
    
    states = ['ESTABLISHED', 'CLOSED', 'LISTEN', 'SYN_SENT', 'FIN_WAIT']
    df['state'] = np.random.choice(states, n_samples)
    
    # Add target variable (label)
    # 0: normal, 1: attack
    df['label'] = np.random.binomial(1, 0.2, n_samples)  # 20% attacks
    
    # Add attack categories
    attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    df['attack_cat'] = df['label'].apply(lambda x: 'normal' if x == 0 else np.random.choice(attack_types[1:]))
    
    # Save reference data
    print(f"Saving reference data to: {reference_path}")
    df.to_csv(reference_path, index=False)
    print(f"Reference data shape: {df.shape}")
    
    return df

def create_test_data():
    """Create test data with simulated drift"""
    reference_path, new_data_path = get_data_paths()
    
    # Create reference data if it doesn't exist
    if not os.path.exists(reference_path):
        print("Reference data not found. Creating it first...")
        reference_data = create_reference_data()
    else:
        print("Loading reference data...")
        reference_data = pd.read_csv(reference_path)
    
    print(f"Reference data shape: {reference_data.shape}")
    
    # Create new data with drift
    print("\nCreating new data with simulated drift...")
    new_data = reference_data.copy()
    
    # 1. Simulate feature drift by modifying some features
    # Modify numerical features
    numerical_cols = new_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols[:5]:  # Modify first 5 numerical columns
        # Add some noise and shift the distribution
        new_data[col] = new_data[col] * (1 + np.random.normal(0.2, 0.1, size=len(new_data)))
        print(f"Modified numerical feature: {col}")
    
    # 2. Simulate target drift by changing label distribution
    if 'label' in new_data.columns:
        # Increase the proportion of one class
        label_counts = new_data['label'].value_counts()
        most_common_label = label_counts.index[0]
        
        # Randomly change some labels
        mask = np.random.random(len(new_data)) < 0.1  # Change 10% of labels
        new_data.loc[mask, 'label'] = most_common_label
        print("\nModified target distribution:")
        print("New label distribution:")
        print(new_data['label'].value_counts(normalize=True))
    
    # 3. Add some missing values
    for col in new_data.columns[:3]:  # Add missing values to first 3 columns
        mask = np.random.random(len(new_data)) < 0.05  # 5% missing values
        new_data.loc[mask, col] = np.nan
        print(f"Added missing values to: {col}")
    
    # 4. Add some outliers
    for col in numerical_cols[:3]:  # Add outliers to first 3 numerical columns
        mask = np.random.random(len(new_data)) < 0.02  # 2% outliers
        new_data.loc[mask, col] = new_data[col].mean() + 5 * new_data[col].std()
        print(f"Added outliers to: {col}")
    
    # Save new data
    print(f"\nSaving new data to: {new_data_path}")
    new_data.to_csv(new_data_path, index=False)
    print("Done!")
    
    # Print summary of changes
    print("\nSummary of changes:")
    print(f"1. Modified {len(numerical_cols[:5])} numerical features")
    print("2. Changed target distribution")
    print(f"3. Added missing values to 3 features")
    print(f"4. Added outliers to 3 numerical features")
    
    return new_data

if __name__ == "__main__":
    create_test_data() 