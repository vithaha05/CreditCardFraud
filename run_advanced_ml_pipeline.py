import sys
import os

# Import original logic
from fraud_detection_FINAL import DataPreprocessor, DataVisualizer, NetworkAnalyzer, plot_combined_fraud_indicators

# Import new ML extensions
from fraud_ml_extensions import run_ml_extension

def main():
    print("\n" + "="*80)
    print("      CREDIT CARD FRAUD DETECTION - ADVANCED ML & GRAPH ANALYTICS")
    print("="*80)

    # 1. Original Pipeline
    preprocessor = DataPreprocessor()
    try:
        raw_data = preprocessor.load_kaggle_dataset('creditcard.csv')
    except FileNotFoundError:
        print("\n[!] Error: creditcard.csv not found!")
        print("Please ensure the dataset is in the current directory.")
        return

    processed_data = preprocessor.preprocess_data()
    
    # Generate Visualizations
    visualizer = DataVisualizer(processed_data)
    visualizer.plot_all_visualizations()

    # Network Analysis
    network_analyzer = NetworkAnalyzer(processed_data)
    network_analyzer.build_network()
    network_analyzer.compute_centrality_and_plot()
    network_analyzer.peer_group_anomaly_detection()

    # 2. Trigger Advanced ML Extension
    print("\n" + "="*80)
    print("      RUNNING ADVANCED MACHINE LEARNING EXTENSIONS")
    print("="*80)
    
    ml_results = run_ml_extension(processed_data, network_analyzer)

    # 3. Final Combined Insights
    print("\n[*] Final Step: Generating Combined Indicators...")
    plot_combined_fraud_indicators(network_analyzer)

    print("\n" + "="*80)
    print("✅ ENHANCED ANALYSIS COMPLETE!")
    print("   - Original Graph Analytics: SAVED")
    print("   - XGBoost Hybrid Model (SMOTE): TRAINED")
    print("   - Isolation Forest (Anomaly): RUN")
    print("   - New artifacts: ml_feature_importance.png")
    print("="*80)

if __name__ == "__main__":
    main()
