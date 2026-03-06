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

## 🐳 Containerized Deployment (Docker)

FrauduLens can be easily containerized and deployed using Docker. This ensures a consistent environment regardless of your local machine configuration.

### 1. Build the Docker Image
Run the following command from the project root to build the container:
```bash
docker build -t fraudulens .
```

### 2. Run the Container
Launch the dashboard on port `8501`:
```bash
docker run -p 8501:8501 fraudulens
```
Once the container is running, access the dashboard at: [http://localhost:8501](http://localhost:8501)

### Docker Configuration
- **Base Image**: `python:3.9-slim` for optimized image size.
- **Port Exposure**: Exposes port `8501` for Streamlit access.
- **Healthcheck**: Monitors dashboard availability for production stability.
- **Persistence**: Excludes large datasets and local logs via `.dockerignore` for efficient image builds.

---
*Production-ready system developed for Resume Excellence.*
