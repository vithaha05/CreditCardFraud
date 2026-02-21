# 📊 Interactive Plotly Dashboard Guide

This guide explains how to use the new interactive Plotly dashboard for Credit Card Fraud Detection.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install Plotly and Dash separately:
```bash
pip install plotly dash
```

### 2. Run the Dashboard

```bash
python dash_app.py
```

The dashboard will start on **http://localhost:8050**

Open your web browser and navigate to that URL.

## 📑 Dashboard Tabs

### 📊 Overview Tab
- **Key Statistics**: Total transactions, fraud count, fraud rate, and total amounts
- **Fraud Distribution**: Interactive bar and pie charts showing normal vs fraudulent transactions
- **Amount Distribution**: Histograms comparing transaction amounts for normal vs fraud
- **Time Distribution**: Analysis of fraud patterns over 24 hours
- **Feature Correlation**: Top 15 features most correlated with fraud

### 🔍 Network Analysis Tab
- **Network Statistics**: Node degree distribution, top nodes, network metrics, edge weights
- **Network Topology**: Interactive graph visualization (zoom, pan, hover for details)
- **Centrality Measures**: Comparison of PageRank, Betweenness, and Degree centrality
- **Anomaly Scores**: Peer group anomaly detection scores by cardholder

### 🎯 Fraud Detection Tab
- **Combined Fraud Heatmap**: Multi-metric risk assessment for top 40 cardholders
- **Fraud Bubble Chart**: Visual representation of fraud clusters with risk scores

### 📈 ML Insights Tab
- **Model Information**: Details about XGBoost and Isolation Forest models
- **Feature Importance**: Visualization of which features the model values most

### 📋 Data Explorer Tab
- **Interactive Data Table**: Filterable and sortable transaction data
- **Fraud Filter**: Filter by fraud status (All/Fraud/Normal)
- **Pagination**: Adjustable rows per page

## 🎨 Interactive Features

All Plotly charts support:
- **Zoom**: Click and drag to zoom, double-click to reset
- **Pan**: Click and drag to pan around
- **Hover**: Hover over data points for detailed information
- **Legend**: Click legend items to show/hide data series
- **Download**: Click the camera icon to download charts as PNG

## 🔧 Technical Details

### Files Created
- `plotly_dashboard.py`: Contains all Plotly visualization functions
- `dash_app.py`: Main Dash web application
- `requirements.txt`: Updated with Plotly and Dash dependencies

### Integration
The dashboard integrates seamlessly with existing code:
- Uses `DataPreprocessor` from `fraud_detection_FINAL.py`
- Uses `NetworkAnalyzer` for network analysis
- Uses `AdvancedMLDetector` from `fraud_ml_extensions.py`

## 📝 Notes

- The dashboard automatically loads `creditcard.csv` on startup
- If the dataset is not found, you'll see an error message with download instructions
- Network analysis and ML models are computed automatically when the dashboard starts
- Processing may take a few moments depending on dataset size

## 🐛 Troubleshooting

**Dashboard won't start:**
- Ensure `creditcard.csv` is in the project directory
- Check that all dependencies are installed: `pip install -r requirements.txt`

**Charts not displaying:**
- Check browser console for errors (F12)
- Ensure JavaScript is enabled in your browser

**Port 8050 already in use:**
- Change the port in `dash_app.py`: `app.run_server(debug=True, port=8051)`

## 🎯 Next Steps

- Customize colors and styling in `plotly_dashboard.py`
- Add more interactive filters in `dash_app.py`
- Export dashboard data using the Data Explorer tab
- Share insights by downloading charts

Enjoy exploring your fraud detection data interactively! 🎉
