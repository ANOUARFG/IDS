import pandas as pd
import os
import requests

def download_dataset():
    """Download the UNSW-NB15 dataset"""
    print("Downloading dataset...")
    
    # Dataset URLs
    urls = {
        'UNSW-NB15_1.csv': 'https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW-NB15_1.csv',
        'UNSW-NB15_2.csv': 'https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW-NB15_2.csv',
        'UNSW-NB15_3.csv': 'https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW-NB15_3.csv',
        'UNSW-NB15_4.csv': 'https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set&files=UNSW-NB15_4.csv'
    }
    
    for filename, url in urls.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                raise
        else:
            print(f"{filename} already exists")

def check_dataset_files():
    """Check the dataset files and their structure"""
    print("Checking dataset files...")
    
    # First, ensure we have the dataset files
    download_dataset()
    
    # Check if files exist
    for i in range(1, 5):
        file_path = f'UNSW-NB15_{i}.csv'
        if os.path.exists(file_path):
            print(f"\nFile {file_path} exists")
            # Try to read first few lines
            try:
                # Read first 5 rows to check structure
                df_sample = pd.read_csv(
                    file_path,
                    nrows=5,
                    low_memory=False,
                    encoding='utf-8',
                    on_bad_lines='skip'
                )
                print(f"Successfully read sample from {file_path}")
                print("Columns:", df_sample.columns.tolist())
                print("Data types:\n", df_sample.dtypes)
                print("Sample data:\n", df_sample.head())
                
                # Check for required columns
                required_columns = ['attack_cat', 'label']
                missing_columns = [col for col in required_columns if col not in df_sample.columns]
                if missing_columns:
                    print(f"WARNING: Missing required columns: {missing_columns}")
                
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
        else:
            print(f"\nFile {file_path} does not exist")
    
    # Check file sizes
    print("\nFile sizes:")
    for i in range(1, 5):
        file_path = f'UNSW-NB15_{i}.csv'
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"{file_path}: {size_mb:.2f} MB")

if __name__ == "__main__":
    check_dataset_files() 