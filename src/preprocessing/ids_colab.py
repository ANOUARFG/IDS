# Install required packages
!pip install numpy pandas scikit-learn matplotlib seaborn joblib h5py requests

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import h5py
import os
import requests
import warnings
warnings.filterwarnings('ignore')

# Create a directory for saving models and components
!mkdir -p saved_models
!mkdir -p saved_components

def download_dataset():
    """Verify the UNSW-NB15 dataset files"""
    print("Checking dataset files...")
    
    # Check if files already exist
    required_files = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("\nMissing dataset files:", missing_files)
        print("\nPlease download the UNSW-NB15 dataset from one of these sources:")
        print("1. Official UNSW website:")
        print("   https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/")
        print("   - Click on 'UNSW-NB15 - CSV Files'")
        print("   - Download and extract the files")
        print("\n2. Kaggle dataset:")
        print("   https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
        print("   - Download the dataset")
        print("   - Extract the files")
        print("\nAfter downloading, place the following files in your working directory:")
        for file in required_files:
            print(f"   - {file}")
        print("\nNote: The dataset is large (about 2.5GB total). Make sure you have enough disk space.")
        raise Exception("Please download the dataset manually and place the files in your working directory")
    
    # Verify existing files
    for filename in required_files:
        print(f"\nVerifying {filename}...")
        try:
            # First, check if the file contains HTML content
            with open(filename, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if '<!DOCTYPE' in first_line or '<html' in first_line:
                    print(f"ERROR: {filename} contains HTML content instead of CSV data.")
                    print("This usually means the file wasn't downloaded correctly.")
                    print("\nPlease follow these steps:")
                    print("1. Delete the existing file:", filename)
                    print("2. Download the dataset from Kaggle:")
                    print("   https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
                    print("3. Extract the files and place them in your working directory")
                    raise Exception(f"{filename} contains HTML content. Please download the correct CSV file.")
                
                # Read first 10 lines to analyze structure
                f.seek(0)  # Go back to start of file
                header = f.readline().strip()
                print("Header line:", header)
                print("Number of columns in header:", len(header.split(',')))
                
                # Read a few more lines to check for inconsistencies
                for j in range(5):
                    line = f.readline().strip()
                    print(f"Line {j+2}: {len(line.split(','))} columns")
            
            # Try different CSV reading parameters
            try:
                # First attempt: standard parameters
                df_sample = pd.read_csv(
                    filename,
                    nrows=5,
                    encoding='utf-8',
                    engine='c',
                    header=None
                )
            except Exception as e1:
                print(f"First attempt failed: {str(e1)}")
                try:
                    # Second attempt: with different encoding
                    df_sample = pd.read_csv(
                        filename,
                        nrows=5,
                        encoding='latin1',
                        engine='c',
                        header=None
                    )
                except Exception as e2:
                    print(f"Second attempt failed: {str(e2)}")
                    # Third attempt: with more lenient parsing
                    df_sample = pd.read_csv(
                        filename,
                        nrows=5,
                        encoding='utf-8',
                        sep=',',
                        on_bad_lines='skip',
                        engine='c',
                        header=None
                    )
            
            print(f"Successfully verified {filename}")
            print("First few rows:")
            print(df_sample.head())
            
        except Exception as e:
            print(f"Error verifying {filename}: {str(e)}")
            print("File content preview:")
            with open(filename, 'r', encoding='utf-8') as f:
                print(f.read()[:500])  # Print first 500 characters
            raise Exception(f"Error verifying {filename}. Please ensure the file is a valid CSV file.")

def load_data():
    """Load and combine all parts of the UNSW-NB15 dataset"""
    print("Loading dataset parts...")
    dfs = []
    
    # Define column names for UNSW-NB15 dataset
    column_names = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
        'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts',
        'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
        'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
        'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
    ]
    
    # Define numeric columns
    numeric_columns = [
        'sport', 'dsport', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
        'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
        'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'sintpkt', 'dintpkt',
        'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
        'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
        'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'label'
    ]
    
    # Define categorical columns
    categorical_columns = ['proto', 'state', 'service', 'attack_cat']
    
    try:
        # First, ensure we have the dataset files
        download_dataset()
        
        for i in range(1, 5):
            print(f"Loading part {i}...")
            filename = f'UNSW-NB15_{i}.csv'
            
            # Try different CSV reading parameters
            try:
                # First attempt: standard parameters with column names
                df = pd.read_csv(
                    filename,
                    encoding='utf-8',
                    engine='c',
                    names=column_names,
                    header=None
                )
            except Exception as e1:
                print(f"First attempt failed: {str(e1)}")
                try:
                    # Second attempt: with different encoding
                    df = pd.read_csv(
                        filename,
                        encoding='latin1',
                        engine='c',
                        names=column_names,
                        header=None
                    )
                except Exception as e2:
                    print(f"Second attempt failed: {str(e2)}")
                    # Third attempt: with more lenient parsing
                    df = pd.read_csv(
                        filename,
                        encoding='utf-8',
                        sep=',',
                        on_bad_lines='skip',
                        engine='c',
                        names=column_names,
                        header=None
                    )
            
            # Convert numeric columns
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle categorical columns
            for col in categorical_columns:
                if col in df.columns:
                    # Fill missing values with mode before converting to categorical
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
                    df[col] = df[col].astype('category')
            
            # Print column names for debugging
            print(f"\nColumns in part {i}:")
            print(df.columns.tolist())
            print("\nData types:")
            print(df.dtypes)
            
            # Check if the file contains HTML content
            if '<!DOCTYPE' in str(df.iloc[0, 0]):
                print(f"WARNING: Part {i} appears to contain HTML content instead of CSV data")
                continue
            
            print(f"Successfully loaded part {i} with shape: {df.shape}")
            dfs.append(df)
        
        if not dfs:
            raise ValueError("No valid data was loaded from any of the files")
        
        # Combine all parts
        df = pd.concat(dfs, ignore_index=True)
        print(f"Total samples loaded: {len(df)}")
        
        # Verify the data
        print("\nVerifying data...")
        print("Columns:", df.columns.tolist())
        print("Data types:\n", df.dtypes)
        print("\nMissing values:\n", df.isnull().sum())
        
        # Check for required columns
        required_columns = ['attack_cat', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"\nWARNING: Missing required columns: {missing_columns}")
            print("Available columns:", df.columns.tolist())
            raise ValueError(f"Required columns {missing_columns} are missing from the dataset")
        
        # Fill missing values in numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def analyze_dataset(df):
    """Analyze the dataset and return important statistics"""
    print("\n=== Dataset Analysis ===")
    
    # Basic information
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    # Check if required columns exist
    required_columns = ['attack_cat', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"\nWARNING: Missing required columns: {missing_columns}")
        print("Available columns:", df.columns.tolist())
        raise ValueError(f"Required columns {missing_columns} are missing from the dataset")
    
    # Target distribution
    print("\nTarget Distribution:")
    target_dist = df['attack_cat'].value_counts()
    print(target_dist)
    
    # Plot target distribution
    plt.figure(figsize=(12, 6))
    target_dist.plot(kind='bar')
    plt.title('Distribution of Attack Categories')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Missing values
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    return target_dist

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
    X = df.drop('attack_cat', axis=1)  # Using attack_cat as target
    y = df['attack_cat']
    
    # Encode target variable
    print("\nEncoding target variable...")
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    
    print("\nEncoding categorical features...")
    for column in categorical_columns:
        print(f"Encoding {column}...")
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column].astype(str))
    
    # Convert all features to numeric type
    X = X.astype(float)
    
    # Feature selection using SelectKBest
    print("\nPerforming feature selection...")
    selector = SelectKBest(f_classif, k='all')
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names and their scores
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    })
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_scores.head(10))
    
    # Scale features using RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Apply PCA for dimensionality reduction
    print("\nApplying PCA...")
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)
    print(f"Reduced dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]}")
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.show()
    
    # Add target encoder to the returned components
    label_encoders['target'] = target_encoder
    
    return X_pca, y, scaler, label_encoders, pca, selector, feature_scores

def save_components(scaler, label_encoders, pca, selector, feature_scores, target_dist):
    """Save all necessary components for local use"""
    print("\nSaving components...")
    
    # Save scaler
    joblib.dump(scaler, 'saved_components/scaler.joblib')
    
    # Save label encoders
    joblib.dump(label_encoders, 'saved_components/label_encoders.joblib')
    
    # Save PCA
    joblib.dump(pca, 'saved_components/pca.joblib')
    
    # Save feature selector
    joblib.dump(selector, 'saved_components/selector.joblib')
    
    # Save feature scores
    feature_scores.to_csv('saved_components/feature_scores.csv', index=False)
    
    # Save target distribution
    target_dist.to_csv('saved_components/target_distribution.csv')
    
    print("Components saved successfully!")

def save_model(mlp, results):
    """Save the trained model and results"""
    print("\nSaving model and results...")
    
    # Save the model
    joblib.dump(mlp, 'saved_models/mlp_model.joblib')
    
    # Save results
    with h5py.File('saved_models/model_results.h5', 'w') as f:
        f.create_dataset('mean_cv_score', data=results['model_performance']['mean_cv_score'])
        f.create_dataset('std_cv_score', data=results['model_performance']['std_cv_score'])
        f.create_dataset('test_score', data=results['model_performance']['test_score'])
        f.create_dataset('pca_components', data=results['pca_components'])
        f.create_dataset('selected_features', data=results['selected_features'])
    
    print("Model and results saved successfully!")

def main():
    # Load data
    df = load_data()
    
    # Analyze dataset
    target_dist = analyze_dataset(df)
    
    # Preprocess data
    X, y, scaler, label_encoders, pca, selector, feature_scores = preprocess_data(df)
    
    # Convert y to numpy array if it's a pandas Series
    if isinstance(y, pd.Series):
        y = y.values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create K-Fold splits
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize MLPClassifier with more stable configuration
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Simpler architecture
        activation='relu',
        solver='adam',
        alpha=0.001,  # Increased L2 penalty
        batch_size=256,  # Fixed batch size
        learning_rate='constant',  # Changed to constant learning rate
        learning_rate_init=0.001,  # Explicit learning rate
        max_iter=1000,  # Reduced max iterations
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,  # Increased patience
        random_state=42,
        verbose=True
    )
    
    # Perform K-Fold cross-validation
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"\nTraining Fold {fold}")
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        try:
            # Train the model
            print("Training model...")
            mlp.fit(X_fold_train, y_fold_train)
            
            # Evaluate
            train_score = mlp.score(X_fold_train, y_fold_train)
            val_score = mlp.score(X_fold_val, y_fold_val)
            
            print(f"Fold {fold} - Training accuracy: {train_score:.4f}")
            print(f"Fold {fold} - Validation accuracy: {val_score:.4f}")
            
            fold_scores.append(val_score)
            
        except Exception as e:
            print(f"Error in fold {fold}: {str(e)}")
            print("Trying with different parameters...")
            
            # Try with different parameters if first attempt fails
            mlp = MLPClassifier(
                hidden_layer_sizes=(50,),  # Even simpler architecture
                activation='relu',
                solver='adam',
                alpha=0.01,  # Further increased L2 penalty
                batch_size=128,  # Smaller batch size
                learning_rate='constant',
                learning_rate_init=0.0005,  # Smaller learning rate
                max_iter=500,  # Further reduced max iterations
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42,
                verbose=True
            )
            
            try:
                mlp.fit(X_fold_train, y_fold_train)
                train_score = mlp.score(X_fold_train, y_fold_train)
                val_score = mlp.score(X_fold_val, y_fold_val)
                
                print(f"Fold {fold} - Training accuracy: {train_score:.4f}")
                print(f"Fold {fold} - Validation accuracy: {val_score:.4f}")
                
                fold_scores.append(val_score)
            except Exception as e2:
                print(f"Second attempt failed: {str(e2)}")
                continue
    
    if not fold_scores:
        raise ValueError("No successful training attempts across all folds")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    y_pred = mlp.predict(X_test)
    test_score = mlp.score(X_test, y_test)
    print(f"Test set accuracy: {test_score:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    print("\nCross-validation results:")
    print(f"Mean validation accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    
    # Save important information for next steps
    results = {
        'feature_importance': feature_scores,
        'target_distribution': target_dist,
        'model_performance': {
            'mean_cv_score': np.mean(fold_scores),
            'std_cv_score': np.std(fold_scores),
            'test_score': test_score
        },
        'pca_components': pca.n_components_,
        'selected_features': len(feature_scores)
    }
    
    print("\n=== Results Summary ===")
    print(f"Number of selected features: {results['selected_features']}")
    print(f"Number of PCA components: {results['pca_components']}")
    print(f"Mean CV accuracy: {results['model_performance']['mean_cv_score']:.4f}")
    print(f"Test accuracy: {results['model_performance']['test_score']:.4f}")
    
    # Save components and model
    save_components(scaler, label_encoders, pca, selector, feature_scores, target_dist)
    save_model(mlp, results)
    
    return results

if __name__ == "__main__":
    results = main() 