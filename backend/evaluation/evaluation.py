import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)
from backend.logger import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    def __init__(self, y_true, y_prob, y_pred, model_name='Model'):
        self.y_true = y_true
        self.y_prob = y_prob
        self.y_pred = y_pred
        self.model_name = model_name

    def display_metrics(self):
        """
        Calculates and logs all relevant metrics for the model.
        """
        logger.info("-" * 40)
        logger.info(f"PERFORMANCE EVALUATION - {self.model_name.upper()}")
        logger.info("-" * 40)
        
        # ROC AUC
        self.roc_auc = roc_auc_score(self.y_true, self.y_prob)
        logger.info(f"  - ROC-AUC Score: {self.roc_auc:.4f}")
        
        # Precision, Recall, F1
        self.precision = precision_score(self.y_true, self.y_pred)
        self.recall = recall_score(self.y_true, self.y_pred)
        self.f1 = f1_score(self.y_true, self.y_pred)
        
        logger.info(f"  - Precision: {self.precision:.4f}")
        logger.info(f"  - Recall: {self.recall:.4f}")
        logger.info(f"  - F1-Score: {self.f1:.4f}")
        
        # Classification Report
        logger.info("  - Detailed Classification Report:")
        report = classification_report(self.y_true, self.y_pred)
        for line in report.split('\n'):
            if line.strip():
                logger.info(f"    {line}")
        
        logger.info("-" * 40)
        
        return {
            'roc_auc': self.roc_auc,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1
        }

    def plot_confusion_matrix(self, save_path=None):
        """
        Saves a heatmap of the confusion matrix.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.ylabel('True Label', fontweight='bold')
        plt.title(f'Confusion Matrix: {self.model_name}', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Confusion matrix saved to: {save_path}")
        plt.close()

    def plot_roc_curve(self, save_path=None):
        """
        Saves the ROC Curve plot.
        """
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_prob)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC curve (area = {self.roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='#34495e', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title(f'ROC Curve: {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"ROC curve saved to: {save_path}")
        plt.close()

def compare_models(metrics_dict_list, model_names):
    """
    Creates a comparison table (log-based) and a plot.
    """
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    # Simple log table
    header = f"{'Model':<25} | {'ROC-AUC':<10} | {'F1-Score':<10} | {'Recall':<10}"
    logger.info(header)
    logger.info("-" * 60)
    for model, metrics in zip(model_names, metrics_dict_list):
        row = f"{model:<25} | {metrics['roc_auc']:<10.4f} | {metrics['f1']:<10.4f} | {metrics['recall']:<10.4f}"
        logger.info(row)
    logger.info("=" * 60)
    
    # Create comparison plot
    metrics_to_plot = ['roc_auc', 'f1', 'precision', 'recall']
    comparison_data = []
    for model, metrics in zip(model_names, metrics_dict_list):
        for m in metrics_to_plot:
            comparison_data.append({
                'Model': model,
                'Metric': m,
                'Value': metrics[m]
            })
    
    df_comp = pd.DataFrame(comparison_data)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Metric', y='Value', hue='Model', data=df_comp, palette='viridis', alpha=0.8, edgecolor='black')
    plt.title('Performance Comparison: Fraud Detection Models', fontsize=16, fontweight='bold')
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', frameon=True)
    plt.savefig('performance_comparison.png', dpi=300)
    plt.close()
    logger.info("Comparison plot saved to: performance_comparison.png")
