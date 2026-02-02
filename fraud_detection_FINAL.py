import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CUSTOM CENTRALITY MEASURES
# ============================================================================

def compute_pagerank(graph, d=0.85, max_iter=100, tol=1e-6):
    nodes = list(graph.nodes())
    N = len(nodes)
    pagerank = {node: 1.0 / N for node in nodes}
    for i in range(max_iter):
        new_pagerank = {}
        for node in nodes:
            rank_sum = 0.0
            for pred in graph.predecessors(node):
                out_deg = graph.out_degree(pred)
                if out_deg > 0:
                    rank_sum += pagerank[pred] / out_deg
            new_pagerank[node] = (1 - d) / N + d * rank_sum
        # check convergence
        if all(abs(new_pagerank[n] - pagerank[n]) < tol for n in nodes):
            break
        pagerank = new_pagerank
    return pagerank

def compute_betweenness_centrality(graph):
    nodes = list(graph.nodes())
    betweenness = {node: 0.0 for node in nodes}
    for s in nodes:
        stack = []
        pred = {w: [] for w in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        dist = dict.fromkeys(nodes, -1)
        sigma[s] = 1.0
        dist[s] = 0
        queue = [s]
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in graph.successors(v):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = dict.fromkeys(nodes, 0.0)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w]) if sigma[w] > 0 else 0
            if w != s:
                betweenness[w] += delta[w]
    n = len(nodes)
    if n > 2:
        for k in betweenness:
            betweenness[k] /= ((n - 1) * (n - 2))
    return betweenness

# ============================================================================
# 1. DATA PREPROCESSING MODULE
# ============================================================================

class DataPreprocessor:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None

    def load_kaggle_dataset(self, csv_path='creditcard.csv'):
        print("[*] Loading Kaggle dataset...")
        self.raw_data = pd.read_csv(csv_path)
        print(f"[✓] Dataset loaded: {self.raw_data.shape[0]} rows, {self.raw_data.shape[1]} columns")
        return self.raw_data

    def preprocess_data(self):
        print("\n[*] Starting data preprocessing...")
        df = self.raw_data.copy()
        print(f"  [Step 1] Checking missing values...")
        missing = df.isnull().sum().sum()
        if missing == 0:
            print("    ✓ No missing values")
        else:
            print(f"    ! Found {missing} missing values - removing rows")
            df = df.dropna()
        print(f"  [Step 2] Dataset statistics:")
        print(f"    Total transactions: {len(df)}")
        print(f"    Fraud transactions: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.2f}%)")
        print(f"    Normal transactions: {len(df) - df['Class'].sum()}")
        print(f"  [Step 3] Amount feature analysis:")
        print(f"    Min: ${df['Amount'].min():.2f}")
        print(f"    Max: ${df['Amount'].max():.2f}")
        print(f"    Mean: ${df['Amount'].mean():.2f}")
        print(f"    Median: ${df['Amount'].median():.2f}")
        df['TimeBucket'] = (df['Time'] // 3600).astype(int)
        df['Cardholder_ID'] = pd.factorize(df['V1'].astype(str) + df['V2'].astype(str))[0] % 500
        df['Channel_ID'] = df['TimeBucket'] % 100
        df['Amount_Category'] = pd.cut(df['Amount'], bins=5, labels=False)
        print(f"  [Step 4] Created network feature columns")
        print(f"    Unique Cardholders: {df['Cardholder_ID'].nunique()}")
        print(f"    Unique Channels: {df['Channel_ID'].nunique()}")
        print("  [✓] Preprocessing complete!\n")
        self.processed_data = df
        return df

# ============================================================================
# 2. VISUALIZATION MODULE
# ============================================================================

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_fraud_distribution(self):
        print("[*] Creating fraud distribution plot...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fraud_counts = self.data['Class'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0].bar(['Normal', 'Fraud'], fraud_counts.values, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
        axes[0].set_title('Transaction Distribution (Normal vs Fraud)', fontsize=14, fontweight='bold')
        for i, v in enumerate(fraud_counts.values):
            axes[0].text(i, v + 5000, str(v), ha='center', fontweight='bold')
        fraud_pct = self.data['Class'].value_counts(normalize=True) * 100
        axes[1].pie(fraud_pct.values, labels=['Normal', 'Fraud'], autopct='%1.2f%%',
                    colors=colors, startangle=90, textprops={'fontsize': 12})
        axes[1].set_title('Fraud Percentage', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('01_fraud_distribution.png', dpi=300, bbox_inches='tight')
        print("  [✓] Saved: 01_fraud_distribution.png")
        plt.close()

    def plot_amount_distribution(self):
        print("[*] Creating amount distribution plot...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        normal_amounts = self.data[self.data['Class'] == 0]['Amount']
        fraud_amounts = self.data[self.data['Class'] == 1]['Amount']
        axes[0].hist(normal_amounts, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[0].set_title('Normal Transactions - Amount Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Amount ($)', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_xlim(0, 500)
        axes[1].hist(fraud_amounts, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1].set_title('Fraudulent Transactions - Amount Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Amount ($)', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_xlim(0, 500)
        plt.tight_layout()
        plt.savefig('02_amount_distribution.png', dpi=300, bbox_inches='tight')
        print("  [✓] Saved: 02_amount_distribution.png")
        plt.close()

    def plot_time_distribution(self):
        print("[*] Creating time distribution plot...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        normal_time = self.data[self.data['Class'] == 0]['Time'].values / 3600
        fraud_time = self.data[self.data['Class'] == 1]['Time'].values / 3600
        axes[0].hist(normal_time, bins=50, color='#2ecc71', alpha=0.7, label='Normal', edgecolor='black')
        axes[0].hist(fraud_time, bins=50, color='#e74c3c', alpha=0.7, label='Fraud', edgecolor='black')
        axes[0].set_title('Transactions Over Time (24 hours)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Hour of Day', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].legend(fontsize=11)
        hourly_fraud = self.data.groupby(self.data['Time'] // 3600)['Class'].agg(['sum', 'count'])
        hourly_fraud['fraud_ratio'] = hourly_fraud['sum'] / hourly_fraud['count'] * 100
        axes[1].plot(hourly_fraud.index, hourly_fraud['fraud_ratio'], marker='o', color='#e74c3c', linewidth=2, markersize=6)
        axes[1].fill_between(hourly_fraud.index, hourly_fraud['fraud_ratio'], alpha=0.3, color='#e74c3c')
        axes[1].set_title('Fraud Rate by Hour of Day', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Hour of Day', fontweight='bold')
        axes[1].set_ylabel('Fraud Ratio (%)', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('03_time_distribution.png', dpi=300, bbox_inches='tight')
        print("  [✓] Saved: 03_time_distribution.png")
        plt.close()

    def plot_feature_correlation(self):
        print("[*] Creating feature correlation heatmap...")
        feature_name_map = {
            'Time': 'Transaction Time',
            'Amount': 'Transaction Amount',
            'Cardholder_ID': 'Cardholder ID',
            'Channel_ID': 'Channel ID',
            'TimeBucket': 'Hour Bucket',
            'Amount_Category': 'Amount Category'
        }
        for i in range(1, 29):
            feature_name_map[f'V{i}'] = f'Principal Component {i}'
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        fraud_correlation = self.data[numeric_cols].corr()['Class'].sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = fraud_correlation[fraud_correlation.index != 'Class'].head(15)
        readable_names = [feature_name_map.get(col, col) for col in top_features.index]
        colors_corr = ['#e74c3c' if x > 0 else '#2ecc71' for x in top_features.values]
        ax.barh(range(len(top_features)), top_features.values, color=colors_corr, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(readable_names)
        ax.set_xlabel('Correlation with Fraud', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Features Correlated with Fraud', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('04_feature_correlation.png', dpi=300, bbox_inches='tight')
        print("  [✓] Saved: 04_feature_correlation.png")
        plt.close()

    def plot_all_visualizations(self):
        print("\n" + "="*70)
        print("GENERATING ALL VISUALIZATIONS")
        print("="*70)
        self.plot_fraud_distribution()
        self.plot_amount_distribution()
        self.plot_time_distribution()
        self.plot_feature_correlation()
        print("\n[✓] All visualizations generated and saved!")

# ============================================================================
# 3. NETWORK ANALYSIS + ADVANCED METHODS
# ============================================================================

class NetworkAnalyzer:
    def __init__(self, data):
        self.data = data
        self.graph = None

    def build_network(self):
        print("\n[*] Building transaction network...")
        self.graph = nx.DiGraph()
        for idx, row in self.data.iterrows():
            cardholder = int(row['Cardholder_ID'])
            channel = int(row['Channel_ID'])
            fraud = int(row['Class'])
            ch_node = f"CH_{cardholder}"
            ch_channel = f"CHAN_{channel}"
            self.graph.add_node(ch_node, type='cardholder')
            self.graph.add_node(ch_channel, type='channel')
            if self.graph.has_edge(ch_node, ch_channel):
                self.graph[ch_node][ch_channel]['weight'] += 1
                self.graph[ch_node][ch_channel]['fraud'] += fraud
            else:
                self.graph.add_edge(ch_node, ch_channel, weight=1, fraud=fraud)
        print(f"[✓] Network built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

    def visualize_network_stats(self):
        print("[*] Creating network statistics plots...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        axes[0, 0].hist(degrees, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Node Degree Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Degree', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        top_degrees = sorted([(n, self.graph.degree(n)) for n in self.graph.nodes()], key=lambda x: x[1], reverse=True)[:10]
        nodes, degs = zip(*top_degrees)
        axes[0, 1].barh(range(len(nodes)), degs, color='#9b59b6', alpha=0.7, edgecolor='black')
        axes[0, 1].set_yticks(range(len(nodes)))
        axes[0, 1].set_yticklabels(nodes, fontsize=8)
        axes[0, 1].set_title('Top 10 Nodes by Degree', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Degree', fontweight='bold')
        density = nx.density(self.graph)
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
        metrics = ['Density', 'Avg Degree', 'Nodes', 'Edges']
        values = [density, avg_degree/10, num_nodes/50, num_edges/50]
        colors_metrics = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
        axes[1, 0].bar(metrics, values, color=colors_metrics, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Network Metrics', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Value (scaled)', fontweight='bold')
        weights = [data['weight'] for u, v, data in self.graph.edges(data=True)]
        axes[1, 1].hist(weights, bins=30, color='#1abc9c', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Edge Weight Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Edge Weight', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontweight='bold')
        plt.tight_layout()
        plt.savefig('05_network_statistics.png', dpi=300, bbox_inches='tight')
        print("  [✓] Saved: 05_network_statistics.png")
        plt.close()

    def visualize_network_topology(self):
        print("[*] Creating network topology visualization...")
        top_nodes = sorted([(n, self.graph.degree(n)) for n in self.graph.nodes()],
                           key=lambda x: x[1], reverse=True)[:30]
        top_nodes = [n for n, d in top_nodes]
        subgraph = self.graph.subgraph(top_nodes)
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        nx.draw_networkx_nodes(subgraph, pos, node_color='#3498db', node_size=300, ax=ax, alpha=0.9)
        nx.draw_networkx_edges(subgraph, pos, edge_color='#95a5a6', arrows=True, arrowsize=15, ax=ax, alpha=0.5,
                              connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_labels(subgraph, pos, font_size=7, ax=ax)
        ax.set_title('Transaction Network Topology (Top 30 Nodes)', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('06_network_topology.png', dpi=300, bbox_inches='tight')
        print("  [✓] Saved: 06_network_topology.png")
        plt.close()

    def compute_centrality_and_plot(self):
        print("[*] Computing centrality measures...")
        pagerank = compute_pagerank(self.graph)
        betweenness = compute_betweenness_centrality(self.graph)
        degree_cent = {node: self.graph.degree(node) / (len(self.graph)-1) if len(self.graph) > 1 else 0.0 for node in self.graph.nodes()}
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        nodes_pr, vals_pr = zip(*top_pr)
        axes[0].barh(range(len(nodes_pr)), vals_pr, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[0].set_yticks(range(len(nodes_pr)))
        axes[0].set_yticklabels(nodes_pr, fontsize=9)
        axes[0].set_title('Top 10 - PageRank', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('PageRank Score', fontweight='bold')
        top_bt = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        nodes_bt, vals_bt = zip(*top_bt)
        axes[1].barh(range(len(nodes_bt)), vals_bt, color='#3498db', alpha=0.7, edgecolor='black')
        axes[1].set_yticks(range(len(nodes_bt)))
        axes[1].set_yticklabels(nodes_bt, fontsize=9)
        axes[1].set_title('Top 10 - Betweenness Centrality', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Betweenness Score', fontweight='bold')
        top_dc = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        nodes_dc, vals_dc = zip(*top_dc)
        axes[2].barh(range(len(nodes_dc)), vals_dc, color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[2].set_yticks(range(len(nodes_dc)))
        axes[2].set_yticklabels(nodes_dc, fontsize=9)
        axes[2].set_title('Top 10 - Degree Centrality', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Degree Centrality Score', fontweight='bold')
        plt.tight_layout()
        plt.savefig('07_centrality_measures.png', dpi=300, bbox_inches='tight')
        print("  [✓] Saved: 07_centrality_measures.png")
        plt.close()
        self.centrality_scores = {'pagerank': pagerank, 'betweenness': betweenness, 'degree': degree_cent}

    def peer_group_anomaly_detection(self):
        print("[*] Performing peer group anomaly detection...")
        df = self.data.copy()
        features = []
        for cardholder in df['Cardholder_ID'].unique():
            ch_data = df[df['Cardholder_ID'] == cardholder]
            feature_dict = {
                'Cardholder_ID': cardholder,
                'num_transactions': len(ch_data),
                'total_amount': ch_data['Amount'].sum(),
                'avg_amount': ch_data['Amount'].mean(),
                'num_channels': ch_data['Channel_ID'].nunique(),
                'fraud_ratio': ch_data['Class'].sum() / len(ch_data)
            }
            features.append(feature_dict)
        feature_df = pd.DataFrame(features)
        feature_df['anomaly_score'] = 0.0
        for col in ['avg_amount', 'num_channels', 'fraud_ratio']:
            mean = feature_df[col].mean()
            std = feature_df[col].std()
            if std > 0:
                feature_df['anomaly_score'] += abs((feature_df[col] - mean) / std) / 3
        feature_df = feature_df.sort_values('anomaly_score', ascending=False)
        print("[✓] Top 10 anomalous cardholders:")
        for i, row in feature_df.head(10).iterrows():
            print(f"    CH_{int(row['Cardholder_ID']):4d} - anomaly: {float(row['anomaly_score']):.4f}, fraud_ratio: {float(row['fraud_ratio']):.2%}")
        self.peer_features = feature_df
        return feature_df

    def plot_peer_anomaly_scores(self):
        df = self.peer_features
        plt.figure(figsize=(14, 6))
        plt.bar(df['Cardholder_ID'], df['anomaly_score'], color='#e74c3c', alpha=0.7)
        plt.title('Peer Group Anomaly Scores by Cardholder', fontweight='bold')
        plt.xlabel('Cardholder ID')
        plt.ylabel('Anomaly Score')
        plt.tight_layout()
        plt.savefig('08_peer_anomaly_scores.png', dpi=300, bbox_inches='tight')
        print("  [✓] Saved: 08_peer_anomaly_scores.png")
        plt.close()

    def snowball_expansion(self, start_nodes, expansion_steps=3):
        print(f"[*] Running snowball expansion from {start_nodes} for {expansion_steps} steps...")
        group = set(start_nodes)
        frontier = set(start_nodes)
        for _ in range(expansion_steps):
            next_frontier = set()
            for node in frontier:
                neighbors = list(self.graph.neighbors(node))
                next_frontier.update(neighbors)
            next_frontier.difference_update(group)
            group.update(next_frontier)
            frontier = next_frontier
        print(f"[✓] Snowball demo: group size = {len(group)}")
        return group

    def plot_snowball_graph(self, group):
        subgraph = self.graph.subgraph(group)
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw(subgraph, pos, with_labels=True, node_color='#3498db', edge_color='#e67e22', node_size=350, font_size=8, alpha=0.7)
        plt.title('Snowball Expansion Subgraph', fontweight='bold')
        plt.tight_layout()
        plt.savefig('09_snowball_expansion.png', dpi=300, bbox_inches='tight')
        print("  [✓] Saved: 09_snowball_expansion.png")
        plt.close()

    def clique_detection(self):
        print("[*] Detecting cliques (fraud rings)...")
        undirected = self.graph.to_undirected()
        cliques = list(nx.find_cliques(undirected))
        significant_cliques = [c for c in cliques if len(c) > 2]
        print(f"[✓] Cliques detected: {len(significant_cliques)}")
        return significant_cliques

    def star_detection(self):
        print("[*] Detecting star patterns (hubs)...")
        stars = []
        for node in self.graph.nodes():
            degree = max(self.graph.out_degree(node), self.graph.in_degree(node))
            if degree > 5:
                stars.append((node, degree))
        stars.sort(key=lambda x: x[1], reverse=True)
        print(f"[✓] Star patterns (hubs): {len(stars)}")
        return stars

    def cycle_detection(self, max_cycles=20):
        print("[*] Detecting cycles/rings...")
        try:
            cycles = list(nx.simple_cycles(self.graph))
            cycles = [c for c in cycles if len(c) > 2][:max_cycles]
            print(f"[✓] Cycles/Rings detected: {len(cycles)}")
        except Exception as e:
            print("[✓] Cycles/Rings detected: 0")
            cycles = []
        return cycles

    def entity_resolution(self):
        print("[*] Running entity resolution (bipartite)...")
        B = nx.Graph()
        for idx, row in self.data.iterrows():
            ch = f"CH_{int(row['Cardholder_ID'])}"
            chan = f"CHAN_{int(row['Channel_ID'])}"
            B.add_node(ch, type='cardholder')
            B.add_node(chan, type='channel')
            B.add_edge(ch, chan)
        components = list(nx.connected_components(B))
        print(f"[✓] Potential duplicate/connected entities (large components):")
        for i, comp in enumerate([c for c in components if len(c) > 10][:5]):
            cardholders = [n for n in comp if 'CH_' in n]
            channels = [n for n in comp if 'CHAN_' in n]
            print(f"    Component {i}: {len(cardholders)} cardholders, {len(channels)} channels")
        return components

# ============================================================================
# 4. FRAUD SUPERGRAPH (COMBINED VISUALIZATION)
# ============================================================================

def plot_combined_fraud_indicators(network_analyzer):
    print("[*] Generating combined fraud indicator plots...")
    df = network_analyzer.peer_features.copy()
    # Add/merge centrality scores
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
    # Compute combined fraud score
    score_cols = ['anomaly_score', 'pagerank_score', 'betweenness_score', 'degree_score', 'in_clique', 'in_snowball']
    df['combined_fraud_score'] = df[score_cols].sum(axis=1)
    df_sort = df.sort_values('combined_fraud_score', ascending=False)
    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_sort[score_cols].head(40), annot=False, cmap='Reds', yticklabels=df_sort['Cardholder_ID'].head(40))
    plt.title('Combined Fraud Risk Indicators (Top 40 Cardholders)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('10_combined_fraud_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    # Bubble plot
    plt.figure(figsize=(16, 8))
    plt.scatter(df_sort['Cardholder_ID'].head(40), df_sort['combined_fraud_score'].head(40),
                s=df_sort['combined_fraud_score'].head(40)*500, c=df_sort['combined_fraud_score'].head(40), cmap='Reds', alpha=0.7)
    plt.xlabel('Cardholder ID')
    plt.ylabel('Aggregate Fraud Risk Score')
    plt.title('Top Fraud Clusters/Bubbles by Combined Indicators')
    plt.tight_layout()
    plt.savefig('11_fraud_bubble_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [✓] Saved: 10_combined_fraud_heatmap.png, 11_fraud_bubble_clusters.png")

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - KAGGLE DATASET VERSION")
    print("="*70)

    preprocessor = DataPreprocessor()
    try:
        raw_data = preprocessor.load_kaggle_dataset('creditcard.csv')
    except FileNotFoundError:
        print("\n[!] Error: creditcard.csv not found!")
        print("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return

    processed_data = preprocessor.preprocess_data()
    visualizer = DataVisualizer(processed_data)
    visualizer.plot_all_visualizations()

    network_analyzer = NetworkAnalyzer(processed_data)
    network_analyzer.build_network()
    network_analyzer.visualize_network_stats()
    network_analyzer.visualize_network_topology()
    network_analyzer.compute_centrality_and_plot()

    # Peer anomaly and viz
    network_analyzer.peer_group_anomaly_detection()
    network_analyzer.plot_peer_anomaly_scores()

    # Snowball & viz
    most_anomalous = network_analyzer.peer_features.iloc[0]['Cardholder_ID']
    start_node = f"CH_{int(most_anomalous)}"
    group = network_analyzer.snowball_expansion([start_node], expansion_steps=3)
    network_analyzer.plot_snowball_graph(group)

    network_analyzer.clique_detection()
    network_analyzer.star_detection()
    network_analyzer.cycle_detection()
    network_analyzer.entity_resolution()

    # Supergraph: Combined fraud visualization!
    plot_combined_fraud_indicators(network_analyzer)

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE! See output images and terminal logs for result details.")
    print("="*70)

if __name__ == "__main__":
    main()