import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        """Initialize the data preprocessor"""
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Define paths relative to project root
        self.components_path = os.path.join(project_root, 'models/components')
        self.processed_data_path = os.path.join(project_root, 'data/processed')
        self.reference_data_path = os.path.join(project_root, 'data/processed/reference_data.csv')
        
        # Ensure directories exist
        os.makedirs(self.components_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Initialize components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.selector = SelectKBest(f_classif, k='all')
        self.label_encoders = {}
    
    def preprocess_data(self, data, is_training=True):
        """Preprocess the data"""
        print("Preprocessing data...")
        
        # Handle missing values
        data = handle_missing_values(data)
        
        # Handle outliers
        data = handle_outliers(data)
        
        # Separate features and target
        X = data.drop('label', axis=1)
        y = data['label']
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            if is_training:
                self.label_encoders[column] = LabelEncoder()
                X[column] = self.label_encoders[column].fit_transform(X[column].astype(str))
            else:
                X[column] = self.label_encoders[column].transform(X[column].astype(str))
        
        # Apply feature selection
        if is_training:
            X_selected = self.selector.fit_transform(X, y)
        else:
            X_selected = self.selector.transform(X)
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X_selected)
        else:
            X_scaled = self.scaler.transform(X_selected)
        
        # Apply PCA
        if is_training:
            X_pca = self.pca.fit_transform(X_scaled)
        else:
            X_pca = self.pca.transform(X_scaled)
        
        return X_pca, y
    
    def save_components(self):
        """Save preprocessing components"""
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.components_path, 'scaler.joblib'))
        
        # Save PCA
        joblib.dump(self.pca, os.path.join(self.components_path, 'pca.joblib'))
        
        # Save feature selector
        joblib.dump(self.selector, os.path.join(self.components_path, 'selector.joblib'))
        
        # Save label encoders
        joblib.dump(self.label_encoders, os.path.join(self.components_path, 'label_encoders.joblib'))
        
        # Save feature scores
        feature_scores = pd.DataFrame({
            'feature': self.selector.feature_names_in_,
            'score': self.selector.scores_
        })
        feature_scores.to_csv(os.path.join(self.components_path, 'feature_scores.csv'), index=False)
    
    def save_processed_data(self, X, y, filename, is_reference=False):
        """Save processed data"""
        processed_data = pd.DataFrame(X)
        processed_data['label'] = y
        processed_data.to_csv(os.path.join(self.processed_data_path, filename), index=False)
        
        # If this is the first batch of data, save it as reference data
        if is_reference and not os.path.exists(self.reference_data_path):
            processed_data.to_csv(self.reference_data_path, index=False)
            print(f"Reference data saved to {self.reference_data_path}")

def load_data():
    """Load and combine all parts of the UNSW-NB15 dataset"""
    print("Loading dataset parts...")
    dfs = []
    for i in range(1, 5):
        df = pd.read_csv(f'data/UNSW-NB15_{i}.csv')
        dfs.append(df)
    
    # Combine all parts
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples loaded: {len(df)}")
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print("Handling missing values...")
    
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Replace inf values with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Handle numerical columns
    numerical_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='median')
    df_clean[numerical_columns] = imputer.fit_transform(df_clean[numerical_columns])
    
    # Handle categorical columns
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
    
    return df_clean

def handle_outliers(df):
    """Handle outliers using IQR method"""
    print("Handling outliers...")
    df_clean = df.copy()
    
    # Only handle numerical columns
    numerical_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
    
    for column in numerical_columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them
        df_clean[column] = df_clean[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean

def preprocess_data(df):
    """Enhanced preprocessing pipeline"""
    print("Preprocessing data...")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column].astype(str))
    
    # Feature selection using SelectKBest
    print("Performing feature selection...")
    selector = SelectKBest(f_classif, k='all')
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Selected features: {len(selected_features)}")
    
    # Scale features using RobustScaler (more robust to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Apply PCA for dimensionality reduction
    print("Applying PCA...")
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)
    print(f"Reduced dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]}")
    
    return X_pca, y, scaler, label_encoders, pca, selector

def create_kfold_splits(X, y, n_splits=5):
    """Create K-Fold cross-validation splits"""
    print(f"Creating {n_splits}-fold cross-validation splits...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return kf.split(X)

def main():
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load raw data
    raw_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                'data/raw/raw_data.csv')
    data = pd.read_csv(raw_data_path)
    
    # Preprocess data
    X_processed, y = preprocessor.preprocess_data(data, is_training=True)
    
    # Save components
    preprocessor.save_components()
    
    # Save processed data and set as reference data
    preprocessor.save_processed_data(X_processed, y, 'processed_data.csv', is_reference=True)

if __name__ == "__main__":
    main() 