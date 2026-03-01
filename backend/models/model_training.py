import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from backend.logger import get_logger

logger = get_logger(__name__)

class FraudModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test, model_dir='models/saved'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_dir = model_dir
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def train_baseline(self, model_type='logistic_regression'):
        """
        Trains a standard ML baseline model.
        """
        logger.info(f"Training Baseline Model: {model_type}...")
        
        if model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=2000)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        else:
            raise ValueError("Invalid model_type. Choose 'logistic_regression' or 'random_forest'.")
            
        model.fit(self.X_train, self.y_train)
        
        # Save model
        save_path = os.path.join(self.model_dir, f"{model_type}_baseline.joblib")
        joblib.dump(model, save_path)
        logger.info(f"Baseline model saved to: {save_path}")
        
        return model

    def train_hybrid_xgb(self, smote_resample=True):
        """
        Trains an XGBoost model, typically using enhanced features (like graph-based ones).
        """
        logger.info("Training Hybrid Model: XGBoost...")
        
        X_train_final, y_train_final = self.X_train, self.y_train
        
        # SMOTE handles class imbalance
        if smote_resample:
            logger.info("  [Step] Applying SMOTE to balance fraud instances...")
            sm = SMOTE(random_state=42)
            X_train_final, y_train_final = sm.fit_resample(self.X_train, self.y_train)
            logger.info(f"  [Step] Balanced Training Set: {X_train_final.shape[0]:,} samples")
            
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42
        )
        
        model.fit(X_train_final, y_train_final)
        
        # Save model
        save_path = os.path.join(self.model_dir, "hybrid_xgb_model.joblib")
        joblib.dump(model, save_path)
        logger.info(f"Hybrid model saved to: {save_path}")
        
        return model

    def run_cross_validation(self, model, name='Model', cv=5):
        """
        Performs K-fold cross-validation and logs results.
        """
        logger.info(f"Running {cv}-fold Cross-Validation for: {name}...")
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='roc_auc')
        
        logger.info(f"Cross-Validation ROC-AUC Results: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
