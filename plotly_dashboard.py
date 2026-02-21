import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx

class PlotlyVisualizer:
    def __init__(self, data):
        self.data = data
    
    def create_fraud_distribution(self):
        """Interactive fraud distribution chart"""
        fraud_counts = self.data['Class'].value_counts()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Transaction Distribution', 'Fraud Percentage'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=['Normal', 'Fraud'],
                y=fraud_counts.values,
                marker_color=['#2ecc71', '#e74c3c'],
                text=fraud_counts.values,
                textposition='outside',
                name='Transactions',
                hovertemplate='Type: %{x}<br>Count: %{y:,}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Pie chart
        fraud_pct = self.data['Class'].value_counts(normalize=True) * 100
        fig.add_trace(
            go.Pie(
                labels=['Normal', 'Fraud'],
                values=fraud_pct.values,
                marker_colors=['#2ecc71', '#e74c3c'],
                textinfo='label+percent',
                hole=0.3,
                hovertemplate='Type: %{label}<br>Percentage: %{percent}<br>Count: %{value:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Fraud Distribution Analysis",
            showlegend=False,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_amount_distribution(self):
        """Interactive amount distribution with hover details"""
        normal_amounts = self.data[self.data['Class'] == 0]['Amount']
        fraud_amounts = self.data[self.data['Class'] == 1]['Amount']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Normal Transactions', 'Fraudulent Transactions')
        )
        
        fig.add_trace(
            go.Histogram(
                x=normal_amounts,
                nbinsx=50,
                marker_color='#2ecc71',
                name='Normal',
                hovertemplate='Amount: $%{x:.2f}<br>Count: %{y}<extra></extra>',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=fraud_amounts,
                nbinsx=50,
                marker_color='#e74c3c',
                name='Fraud',
                hovertemplate='Amount: $%{x:.2f}<br>Count: %{y}<extra></extra>',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Amount ($)", range=[0, 500], row=1, col=1)
        fig.update_xaxes(title_text="Amount ($)", range=[0, 500], row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_layout(
            title_text="Transaction Amount Distribution",
            height=500,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def create_time_distribution(self):
        """Interactive time-based fraud analysis"""
        normal_time = self.data[self.data['Class'] == 0]['Time'].values / 3600
        fraud_time = self.data[self.data['Class'] == 1]['Time'].values / 3600
        
        hourly_fraud = self.data.groupby(self.data['Time'] // 3600)['Class'].agg(['sum', 'count'])
        hourly_fraud['fraud_ratio'] = hourly_fraud['sum'] / hourly_fraud['count'] * 100
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Transactions Over Time', 'Fraud Rate by Hour')
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=normal_time,
                name='Normal',
                marker_color='#2ecc71',
                opacity=0.7,
                nbinsx=50,
                hovertemplate='Hour: %{x:.1f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=fraud_time,
                name='Fraud',
                marker_color='#e74c3c',
                opacity=0.7,
                nbinsx=50,
                hovertemplate='Hour: %{x:.1f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Line chart
        fig.add_trace(
            go.Scatter(
                x=hourly_fraud.index,
                y=hourly_fraud['fraud_ratio'],
                mode='lines+markers',
                fill='tonexty',
                marker_color='#e74c3c',
                name='Fraud Rate',
                hovertemplate='Hour: %{x}<br>Fraud Rate: %{y:.2f}%<extra></extra>',
                line=dict(width=3)
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Fraud Ratio (%)", row=1, col=2)
        
        fig.update_layout(
            title_text="Time-Based Fraud Analysis",
            height=500,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_feature_correlation(self):
        """Interactive feature correlation chart"""
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        fraud_correlation = self.data[numeric_cols].corr()['Class'].sort_values(ascending=False)
        top_features = fraud_correlation[fraud_correlation.index != 'Class'].head(15)
        
        colors_corr = ['#e74c3c' if x > 0 else '#2ecc71' for x in top_features.values]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                y=top_features.index,
                x=top_features.values,
                orientation='h',
                marker_color=colors_corr,
                text=[f'{val:.3f}' for val in top_features.values],
                textposition='outside',
                hovertemplate='Feature: %{y}<br>Correlation: %{x:.4f}<extra></extra>'
            )
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title="Top 15 Features Correlated with Fraud",
            xaxis_title="Correlation with Fraud",
            yaxis_title="Features",
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def create_network_topology(self, graph, top_n=30):
        """Interactive network graph visualization"""
        top_nodes = sorted([(n, graph.degree(n)) for n in graph.nodes()],
                          key=lambda x: x[1], reverse=True)[:top_n]
        top_nodes = [n for n, d in top_nodes]
        subgraph = graph.subgraph(top_nodes)
        
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            degree = subgraph.degree(node)
            node_info.append(f'Node: {node}<br>Degree: {degree}')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                size=15,
                color=[subgraph.degree(n) for n in subgraph.nodes()],
                colorbar=dict(
                    thickness=15,
                    title="Node Degree",
                    xanchor="left",
                    titleside="right"
                ),
                line_width=2,
                line_color='white'
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f'Transaction Network Topology (Top {top_n} Nodes)',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="Interactive Network Graph - Hover for details, drag to pan, zoom with mouse wheel",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="#888", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700,
                template='plotly_white'
            )
        )
        
        return fig
    
    def create_centrality_comparison(self, centrality_scores):
        """Compare different centrality measures"""
        pagerank = centrality_scores.get('pagerank', {})
        betweenness = centrality_scores.get('betweenness', {})
        degree = centrality_scores.get('degree', {})
        
        top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        top_bt = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        top_dc = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('PageRank', 'Betweenness Centrality', 'Degree Centrality')
        )
        
        nodes_pr, vals_pr = zip(*top_pr) if top_pr else ([], [])
        fig.add_trace(
            go.Bar(
                y=list(nodes_pr),
                x=list(vals_pr),
                orientation='h',
                marker_color='#e74c3c',
                name='PageRank',
                hovertemplate='Node: %{y}<br>Score: %{x:.6f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        nodes_bt, vals_bt = zip(*top_bt) if top_bt else ([], [])
        fig.add_trace(
            go.Bar(
                y=list(nodes_bt),
                x=list(vals_bt),
                orientation='h',
                marker_color='#3498db',
                name='Betweenness',
                hovertemplate='Node: %{y}<br>Score: %{x:.6f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        nodes_dc, vals_dc = zip(*top_dc) if top_dc else ([], [])
        fig.add_trace(
            go.Bar(
                y=list(nodes_dc),
                x=list(vals_dc),
                orientation='h',
                marker_color='#2ecc71',
                name='Degree',
                hovertemplate='Node: %{y}<br>Score: %{x:.6f}<extra></extra>'
            ),
            row=1, col=3
        )
        
        fig.update_xaxes(title_text="Score", row=1, col=1)
        fig.update_xaxes(title_text="Score", row=1, col=2)
        fig.update_xaxes(title_text="Score", row=1, col=3)
        
        fig.update_layout(
            title_text="Centrality Measures Comparison",
            height=500,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def create_anomaly_scores(self, peer_features):
        """Interactive anomaly scores visualization"""
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=peer_features['Cardholder_ID'],
                y=peer_features['anomaly_score'],
                marker_color=peer_features['anomaly_score'],
                marker_colorscale='Reds',
                hovertemplate='Cardholder: CH_%{x}<br>Anomaly Score: %{y:.4f}<extra></extra>',
                text=[f'{score:.3f}' for score in peer_features['anomaly_score']],
                textposition='outside'
            )
        )
        
        fig.update_layout(
            title="Peer Group Anomaly Scores by Cardholder",
            xaxis_title="Cardholder ID",
            yaxis_title="Anomaly Score",
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_combined_fraud_heatmap(self, network_analyzer):
        """Interactive heatmap of combined fraud indicators"""
        df = network_analyzer.peer_features.copy()
        
        # Add centrality scores
        for k, scores in network_analyzer.centrality_scores.items():
            key = f"{k}_score"
            df[key] = df['Cardholder_ID'].apply(lambda cid: scores.get(f'CH_{int(cid)}', 0))
        
        # Add clique memberships
        clique_members = set(c for clique in network_analyzer.clique_detection() for c in clique)
        df['in_clique'] = df['Cardholder_ID'].apply(lambda cid: 1 if f'CH_{int(cid)}' in clique_members else 0)
        
        # Add snowball group members
        most_anomalous = df.iloc[0]['Cardholder_ID']
        snowball_group = network_analyzer.snowball_expansion([f'CH_{int(most_anomalous)}'], expansion_steps=3)
        df['in_snowball'] = df['Cardholder_ID'].apply(lambda cid: 1 if f'CH_{int(cid)}' in snowball_group else 0)
        
        score_cols = ['anomaly_score', 'pagerank_score', 'betweenness_score', 'degree_score', 'in_clique', 'in_snowball']
        df['combined_fraud_score'] = df[score_cols].sum(axis=1)
        df_sort = df.sort_values('combined_fraud_score', ascending=False)
        
        fig = go.Figure(data=go.Heatmap(
            z=df_sort[score_cols].head(40).T.values,
            x=[f"CH_{int(cid)}" for cid in df_sort['Cardholder_ID'].head(40)],
            y=score_cols,
            colorscale='Reds',
            hovertemplate='Cardholder: %{x}<br>Metric: %{y}<br>Value: %{z:.4f}<extra></extra>',
            colorbar=dict(title="Risk Score")
        ))
        
        fig.update_layout(
            title="Combined Fraud Risk Indicators (Top 40 Cardholders)",
            xaxis_title="Cardholder ID",
            yaxis_title="Risk Indicators",
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_fraud_bubble_chart(self, network_analyzer):
        """Interactive bubble chart showing fraud clusters"""
        df = network_analyzer.peer_features.copy()
        
        for k, scores in network_analyzer.centrality_scores.items():
            key = f"{k}_score"
            df[key] = df['Cardholder_ID'].apply(lambda cid: scores.get(f'CH_{int(cid)}', 0))
        
        clique_members = set(c for clique in network_analyzer.clique_detection() for c in clique)
        df['in_clique'] = df['Cardholder_ID'].apply(lambda cid: 1 if f'CH_{int(cid)}' in clique_members else 0)
        
        most_anomalous = df.iloc[0]['Cardholder_ID']
        snowball_group = network_analyzer.snowball_expansion([f'CH_{int(most_anomalous)}'], expansion_steps=3)
        df['in_snowball'] = df['Cardholder_ID'].apply(lambda cid: 1 if f'CH_{int(cid)}' in snowball_group else 0)
        
        score_cols = ['anomaly_score', 'pagerank_score', 'betweenness_score', 'degree_score', 'in_clique', 'in_snowball']
        df['combined_fraud_score'] = df[score_cols].sum(axis=1)
        df_sort = df.sort_values('combined_fraud_score', ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df_sort['Cardholder_ID'].head(40),
                y=df_sort['combined_fraud_score'].head(40),
                mode='markers',
                marker=dict(
                    size=df_sort['combined_fraud_score'].head(40) * 50,
                    color=df_sort['combined_fraud_score'].head(40),
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Fraud Score"),
                    line=dict(width=2, color='darkred')
                ),
                text=[f"CH_{int(cid)}" for cid in df_sort['Cardholder_ID'].head(40)],
                hovertemplate='Cardholder: %{text}<br>Fraud Score: %{y:.4f}<br>Cardholder ID: %{x}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title="Top Fraud Clusters by Combined Indicators",
            xaxis_title="Cardholder ID",
            yaxis_title="Aggregate Fraud Risk Score",
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_network_statistics(self, graph):
        """Create network statistics visualizations"""
        degrees = [graph.degree(n) for n in graph.nodes()]
        top_degrees = sorted([(n, graph.degree(n)) for n in graph.nodes()], 
                            key=lambda x: x[1], reverse=True)[:10]
        
        density = nx.density(graph)
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
        
        weights = [data['weight'] for u, v, data in graph.edges(data=True)]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Node Degree Distribution', 'Top 10 Nodes by Degree', 
                          'Network Metrics', 'Edge Weight Distribution')
        )
        
        # Degree distribution
        fig.add_trace(
            go.Histogram(x=degrees, nbinsx=30, marker_color='#3498db', name='Degrees'),
            row=1, col=1
        )
        
        # Top degrees
        if top_degrees:
            nodes, degs = zip(*top_degrees)
            fig.add_trace(
                go.Bar(y=list(nodes), x=list(degs), orientation='h', marker_color='#9b59b6'),
                row=1, col=2
            )
        
        # Network metrics
        metrics = ['Density', 'Avg Degree', 'Nodes', 'Edges']
        values = [density, avg_degree/10, num_nodes/50, num_edges/50]
        colors_metrics = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
        fig.add_trace(
            go.Bar(x=metrics, y=values, marker_color=colors_metrics),
            row=2, col=1
        )
        
        # Edge weights
        fig.add_trace(
            go.Histogram(x=weights, nbinsx=30, marker_color='#1abc9c'),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Degree", row=1, col=1)
        fig.update_xaxes(title_text="Degree", row=1, col=2)
        fig.update_xaxes(title_text="Metric", row=2, col=1)
        fig.update_xaxes(title_text="Edge Weight", row=2, col=2)
        
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Node", row=1, col=2)
        fig.update_yaxes(title_text="Value (scaled)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        fig.update_layout(
            title_text="Network Statistics",
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
