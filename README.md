# 🔍 FrauduLens: Graph-Based Fraud Detection System

**FrauduLens** is a production-grade fraud detection system that combines traditional machine learning with graph-based anomaly detection. By modeling transactions as complex connections in a network, the system identifies sophisticated fraud rings that standard tabular models often miss.

---

## 🏗️ Project Architecture

The project is structured with a clean separation between the backend logic and the interactive UI.

```text
FrauduLens/
├── backend/            # Core Business Logic
│   ├── data/           # Ingestion & Preprocessing
│   ├── graph/          # Graph construction & Anomaly Detection
│   ├── models/         # ML Training (XGBoost, Logistic Regression)
│   ├── evaluation/     # Metrics (ROC-AUC, Precision/Recall)
│   └── logger.py       # Comprehensive Logging
├── ui/                 # Streamlit UI Components
│   └── visualizations.py # Plotly interactive charts
└── app.py              # Main Streamlit Application (Entry Point)
```

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Start the interactive dashboard locally:
```bash
streamlit run app.py
```

### 3. Usage
- **Upload Dataset**: Upload `creditcard.csv` or any compatible transaction CSV via the sidebar.
- **Select Model**: Choose between a baseline Logistic Regression or the Hybrid Graph-ML model.
- **Analyze**: Visualize real-time graph statistics, node degree distributions, and interactive network topologies.
- **Export**: Download performance metrics and predictions as CSV directly from the dashboard.

---

## 🧪 Graph-Based Fraud Insights
Transactions are modeled as a **Directed Graph**:
- **Nodes**: Users (Cardholders) and Merchant Channels.
- **Edges**: Individual transactions between entities.
- **Centrality**: Uses PageRank and Betweenness Centrality to identify "bridge" entities often used in money laundering.

### Advanced Features
- **Clique Detection**: Automated identification of dense fraud rings.
- **Snowball Expansion**: Trace related fraudulent accounts from a single suspicious seed.
- **Star Pattern Detection**: Identify high-frequency "hub" accounts used for card testing.

---

## 📈 Model Performance Highlights

| Model | ROC-AUC | F1-Score | Recall (Fraud) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | 0.9550 | 0.7528 | 0.6837 |
| **Hybrid (XGBoost + Graph)** | **0.9772** | **0.6540** | **0.8776** |

*The Hybrid model catches ~20% more fraud cases by leveraging graph-based features.*

---

## ☁️ Deployment (Streamlit Cloud)
To deploy this project to Streamlit Cloud:
1. Push this repository to GitHub.
2. Connect your GitHub account to [Streamlit Cloud](https://streamlit.io/cloud).
3. Select this repo and `app.py` as the main file.
4. Ensure `creditcard.csv` is either included in the repo or uploaded via the UI.

---
*Production-ready system developed for Resume Excellence.*
