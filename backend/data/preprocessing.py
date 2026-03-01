import pandas as pd
import numpy as np
import os
from backend.logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self, csv_path='creditcard.csv'):
        self.csv_path = csv_path
        self.raw_data = None
        self.processed_data = None

    def load_data(self):
        """
        Loads the credit card fraud dataset from CSV.
        """
        if not os.path.exists(self.csv_path):
            logger.error(f"Dataset file '{self.csv_path}' not found.")
            raise FileNotFoundError(f"'{self.csv_path}' is missing.")
            
        logger.info(f"Loading dataset from: {self.csv_path}...")
        self.raw_data = pd.read_csv(self.csv_path)
        
        # Logging dataset scale
        num_transactions = len(self.raw_data)
        num_fraud = self.raw_data['Class'].sum()
        num_normal = num_transactions - num_fraud
        fraud_ratio = (num_fraud / num_transactions) * 100
        
        logger.info("-" * 40)
        logger.info(f"DATASET SCALE SUMMARY:")
        logger.info(f"  - Total Transactions: {num_transactions:,}")
        logger.info(f"  - Fraud Cases: {num_fraud:,} ({fraud_ratio:.4f}%)")
        logger.info(f"  - Normal Cases: {num_normal:,}")
        
        # In this dataset, V1-V2 are used for user identification features in the original project
        # In a real scenario, we would have explicit user/device IDs.
        # We simulate them to demonstrate graph capabilities.
        unique_users = len(self.raw_data.groupby(['V1', 'V2'])) 
        logger.info(f"  - Unique Potential Users (simulated): {unique_users:,}")
        logger.info("-" * 40)
        
        return self.raw_data

    def load_kaggle_dataset(self, csv_path=None):
        if csv_path: self.csv_path = csv_path
        return self.load_data()

    def load_from_dataframe(self, df):
        """
        Loads data directly from a pandas DataFrame (e.g., from Streamlit upload).
        """
        logger.info("Loading dataset from DataFrame...")
        self.raw_data = df
        return self.raw_data

    def preprocess(self):
        """
        Cleans data and engineers network features.
        """
        if self.raw_data is None:
            self.load_data()
            
        logger.info("Starting data preprocessing...")
        df = self.raw_data.copy()
        
        # Missing values check
        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"Found {missing} missing values. Removing affected rows.")
            df = df.dropna()
        else:
            logger.info("No missing values detected.")
            
        # Amount range statistics
        logger.info(f"Amount Range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f} (Mean: ${df['Amount'].mean():.2f})")
        
        # Feature Engineering for Graph Construction
        # We create TimeBuckets to represent temporal connectivity
        df['TimeBucket'] = (df['Time'] // 3600).astype(int)
        
        # Simulate Cardholder and Channel IDs (as in the original code for demonstration)
        # Using V1 and V2 as a proxy for cardholder identity
        df['Cardholder_ID'] = pd.factorize(df['V1'].astype(str) + df['V2'].astype(str))[0] % 500
        df['Channel_ID'] = df['TimeBucket'] % 100
        df['Amount_Category'] = pd.cut(df['Amount'], bins=5, labels=False)
        
        logger.info(f"Engineered network columns: Cardholder_ID ({df['Cardholder_ID'].nunique()} unique), Channel_ID ({df['Channel_ID'].nunique()} unique)")
        logger.info("Preprocessing complete.")
        
        self.processed_data = df
        return df

    def preprocess_data(self): return self.preprocess()

    def get_train_test_split(self, test_size=0.2, random_state=42):
        """
        Simple split for baseline models.
        """
        from sklearn.model_selection import train_test_split
        
        if self.processed_data is None:
            self.preprocess()
            
        X = self.processed_data.drop(['Class', 'TimeBucket', 'Cardholder_ID', 'Channel_ID', 'Amount_Category'], axis=1)
        y = self.processed_data['Class']
        
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
