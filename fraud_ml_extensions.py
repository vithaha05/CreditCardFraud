import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedMLDetector:
    def __init__(self, data, network_analyzer):
        self.data = data
        self.network_analyzer = network_analyzer
        self.model = None
        self.anomaly_detector = None
        self.features = None
        self.target = 'Class'

    def prepare_hybrid_features(self):
        """
        Combines original tabular features with graph-based centrality scores.
        """
        print("[*] Preparing hybrid features (tabular + graph)...")
        df = self.data.copy()
        
        # Get centrality scores from the network analyzer
        pagerank = self.network_analyzer.centrality_scores.get('pagerank', {})
        betweenness = self.network_analyzer.centrality_scores.get('betweenness', {})
        degree = self.network_analyzer.centrality_scores.get('degree', {})
        
        # Map graph scores back to the dataframe
        df['pagerank_score'] = df['Cardholder_ID'].apply(lambda cid: pagerank.get(f'CH_{int(cid)}', 0))
        df['betweenness_score'] = df['Cardholder_ID'].apply(lambda cid: betweenness.get(f'CH_{int(cid)}', 0))
        df['degree_score'] = df['Cardholder_ID'].apply(lambda cid: degree.get(f'CH_{int(cid)}', 0))
        
        # Select features: V1-V28 + Amount + Time + Graph Features
        v_cols = [f'V{i}' for i in range(1, 29)]
        ml_cols = v_cols + ['Amount', 'Time', 'pagerank_score', 'betweenness_score', 'degree_score']
        
        self.features = df[ml_cols]
        self.labels = df[self.target]
        print(f"    [✓] Feature matrix shape: {self.features.shape}")
        return self.features, self.labels

    def train_hybrid_model(self):
        """
        Trains an XGBoost classifier on the hybrid feature set with SMOTE balancing.
        """
        print("\n[*] Training Hybrid Supervised Model (XGBoost)...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, stratify=self.labels, random_state=42
        )
        
        # Address imbalance using SMOTE
        print("    [Step 1] Applying SMOTE to balance training data...")
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"    [Step 2] Balanced training set: {X_res.shape[0]} samples")
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        self.model.fit(X_res, y_res)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        print("\n[Hybrid Model Results]")
        print(classification_report(y_test, y_pred))
        print(f"Average Precision (AUPRC): {average_precision_score(y_test, y_prob):.4f}")
        
        self.plot_feature_importance()
        return self.model

    def train_unsupervised_anomaly(self):
        """
        Uses Isolation Forest to detect 'zero-day' fraud (outliers).
        """
        print("\n[*] Running Unsupervised Anomaly Detection (Isolation Forest)...")
        # Focus on a subset of features for anomaly detection
        iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        # We only use training data for fitting in a real scenario, but here we flag the whole set
        anomalies = iso_forest.fit_predict(self.features)
        
        # Convert -1 (anomaly) and 1 (normal) to 0 and 1
        self.data['anomaly_flag'] = [1 if x == -1 else 0 for x in anomalies]
        num_anomalies = self.data['anomaly_flag'].sum()
        print(f"    [✓] Flagged {num_anomalies} transactions as anomalies.")
        
        return self.data['anomaly_flag']

    def plot_feature_importance(self):
        """
        Visualizes which features (including graph ones) the model values most.
        """
        importances = pd.Series(self.model.feature_importances_, index=self.features.columns)
        top_10 = importances.sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10.values, y=top_10.index, palette='viridis')
        plt.title('Top 10 Features (Hybrid Model Importance)', fontweight='bold')
        plt.xlabel('F-Score / Importance')
        plt.tight_layout()
        plt.savefig('ml_feature_importance.png', dpi=300)
        print("    [✓] Saved: ml_feature_importance.png")
        plt.close()

def run_ml_extension(processed_data, network_analyzer):
    ml_detector = AdvancedMLDetector(processed_data, network_analyzer)
    ml_detector.prepare_hybrid_features()
    ml_detector.train_hybrid_model()
    ml_detector.train_unsupervised_anomaly()
    return ml_detector
