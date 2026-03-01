import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Ensure backend and ui are reachable
sys.path.append(os.getcwd())

# Import Backend Modules
from backend.data.preprocessing import DataPreprocessor
from backend.graph.graph_builder import GraphBuilder
from backend.models.model_training import FraudModelTrainer
from backend.evaluation.evaluation import ModelEvaluator
from ui.visualizations import PlotlyVisualizer

# Page Configuration
st.set_page_config(
    page_title="FrauduLens – Graph-Based Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

# --- 1. SIDEBAR ---
st.sidebar.title("🔍 FrauduLens Control")
st.sidebar.markdown("Configure and run your fraud detection pipeline.")

uploaded_file = st.sidebar.file_uploader("Upload Transaction Dataset (CSV)", type="csv")

model_choice = st.sidebar.selectbox(
    "Select Detection Model",
    ["Baseline ML (Logistic Regression)", "Hybrid Graph-ML (XGBoost + Graph)"]
)

run_button = st.sidebar.button("Train & Evaluate")

st.sidebar.markdown("---")
st.sidebar.info("FrauduLens uses NetworkX for graph anomaly detection and XGBoost for classification.")

# --- 2. MAIN HEADER ---
st.title("🛡️ FrauduLens – Graph-Based Fraud Detection Dashboard")
st.markdown("""
Welcome to **FrauduLens**. This system leverages **graph theory** and **machine learning** to identify 
suspicious transaction patterns that traditional systems might miss.
""")

# --- 3. DATA INGESTION & PIPELINE ---
@st.cache_data
def process_data(df):
    preprocessor = DataPreprocessor()
    preprocessor.load_from_dataframe(df)
    processed_df = preprocessor.preprocess()
    return processed_df, preprocessor

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    processed_df, preprocessor = process_data(df_raw)
    
    viz = PlotlyVisualizer(processed_df)
    
    # --- 4. GRAPH MODELING ---
    graph_builder = GraphBuilder(processed_df)
    graph_builder.build_network()
    graph_stats = graph_builder.get_graph_stats()
    
    # Quick metrics row
    st.subheader("📊 Dataset & Graph Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    num_total = len(processed_df)
    num_fraud = processed_df['Class'].sum()
    fraud_ratio = (num_fraud / num_total) * 100
    
    col1.metric("Total Transactions", f"{num_total:,}")
    col2.metric("Fraud Cases", f"{num_fraud:,}")
    col3.metric("Fraud Ratio", f"{fraud_ratio:.4f}%")
    col4.metric("Graph Density", f"{graph_stats['edge_count'] / (graph_stats['node_count']**2) if graph_stats['node_count'] > 0 else 0:.4f}")

    # Tabs for better layout
    tab1, tab2, tab3 = st.tabs(["🔍 Exploratory Analysis", "🕸️ Graph Insights", "📈 Detection Results"])
    
    with tab1:
        st.subheader("Distribution Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.create_fraud_distribution(), use_container_width=True)
        with c2:
            st.plotly_chart(viz.create_amount_distribution(), use_container_width=True)
            
        st.plotly_chart(viz.create_time_distribution(), use_container_width=True)
        st.plotly_chart(viz.create_feature_correlation(), use_container_width=True)

    with tab2:
        st.subheader("Network Topology & Statistics")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(viz.create_network_topology(graph_builder.graph), use_container_width=True)
        with c2:
            st.write("**Graph Statistics**")
            st.json(graph_stats)
            st.plotly_chart(viz.create_network_statistics(graph_builder.graph), use_container_width=True)
            
        st.markdown("**Advanced Centrality Measures**")
        centrality_scores = graph_builder.compute_centrality()
        st.plotly_chart(viz.create_centrality_comparison(centrality_scores), use_container_width=True)

    with tab3:
        if run_button:
            with st.spinner("Training model and calculating metrics..."):
                # Prepare features for the selected model
                X_train, X_test, y_train, y_test = preprocessor.get_train_test_split()
                
                if "Hybrid" in model_choice:
                    # Enrich features with centrality
                    centrality_scores = graph_builder.centrality_scores
                    # We need the full dataframe to map scores back correctly
                    # For demonstration, we'll map them to the original processed indices
                    for df_fold in [X_train, X_test]:
                        # Cardholder_ID is needed. Let's retrieve it from the original processed data
                        c_ids = processed_df.loc[df_fold.index, 'Cardholder_ID']
                        df_fold['pagerank_score'] = c_ids.apply(lambda cid: centrality_scores['pagerank'].get(f'CH_{int(cid)}', 0))
                        df_fold['betweenness_score'] = c_ids.apply(lambda cid: centrality_scores['betweenness'].get(f'CH_{int(cid)}', 0))
                        df_fold['degree_score'] = c_ids.apply(lambda cid: centrality_scores['degree'].get(f'CH_{int(cid)}', 0))
                    
                    trainer = FraudModelTrainer(X_train, y_train, X_test, y_test)
                    model = trainer.train_hybrid_xgb()
                    model_name = "XGBoost Hybrid"
                else:
                    trainer = FraudModelTrainer(X_train, y_train, X_test, y_test)
                    model = trainer.train_baseline('logistic_regression')
                    model_name = "Logistic Regression Baseline"
                
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                
                evaluator = ModelEvaluator(y_test, y_prob, y_pred, model_name)
                metrics = evaluator.display_metrics()
                
                # Display Results
                st.success(f"Model Training Complete: {model_name}")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                m2.metric("Precision", f"{metrics['precision']:.4f}")
                m3.metric("Recall", f"{metrics['recall']:.4f}")
                m4.metric("F1-Score", f"{metrics['f1']:.4f}")
                
                # Confusion Matrix Heatmap
                st.subheader("Confusion Matrix")
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                import matplotlib.pyplot as plt
                
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Normal', 'Predicted Fraud'],
                    y=['Actual Normal', 'Actual Fraud'],
                    colorscale='Reds',
                    text=cm,
                    texttemplate="%{text}",
                    showscale=False
                ))
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # ROC Curve
                from sklearn.metrics import roc_curve
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC AUC {metrics["roc_auc"]:.2f}'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), showlegend=False))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig_roc, use_container_width=True)

                # Download Buttons
                st.subheader("📥 Download Results")
                results_df = pd.DataFrame([metrics])
                st.download_button(
                    label="Download Metrics CSV",
                    data=results_df.to_csv(index=False),
                    file_name="fraudu_metrics.csv",
                    mime="text/csv"
                )
                
                preds_df = pd.DataFrame({'y_true': y_test, 'y_prob': y_prob, 'y_pred': y_pred})
                st.download_button(
                    label="Download Predictions CSV",
                    data=preds_df.to_csv(index=False),
                    file_name="fraudu_predictions.csv",
                    mime="text/csv"
                )
        else:
            st.info("Configure settings in the sidebar and click 'Train & Evaluate' to see model performance.")

else:
    st.warning("Please upload a CSV file to get started.")
    st.info("The sample dataset 'creditcard.csv' can be used if you host it locally.")
    if os.path.exists('creditcard.csv'):
        if st.button("Load local 'creditcard.csv'"):
            st.rerun()

# --- 5. FOOTER ---
st.markdown("---")
st.caption("FrauduLens v2.0 | Production-Grade Graph Anomaly Detection")
