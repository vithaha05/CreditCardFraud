import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from backend.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# CUSTOM CENTRALITY MEASURES (Optimized)
# ============================================================================

def compute_pagerank(graph, d=0.85, max_iter=100, tol=1e-6):
    """
    Computes PageRank for nodes in the transaction network.
    Higher PageRank for nodes that interact with many high-degree nodes.
    """
    nodes = list(graph.nodes())
    N = len(nodes)
    if N == 0:
        return {}
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
        # Check convergence
        if all(abs(new_pagerank[n] - pagerank[n]) < tol for n in nodes):
            break
        pagerank = new_pagerank
    return pagerank

def compute_betweenness_centrality(graph):
    """
    Computes Betweenness Centrality.
    High betweenness indicates nodes that act as bridges in the network.
    """
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
# GRAPH BUILDER MODULE
# ============================================================================

class GraphBuilder:
    def __init__(self, data):
        self.data = data
        self.graph = None
        self.centrality_scores = {}

    def build_network(self):
        """
        Constructs a transaction network where:
        - Nodes represent Entities (Cardholders, Channel IDs, Merchant proxies)
        - Edges represent Transactions
        - Weights represent cumulative transaction counts
        """
        logger.info("Building transaction network...")
        self.graph = nx.DiGraph()
        
        # We model the graph as a Bipartite/Multi-partite projection
        # Cardholder_ID -> Channel_ID (a channel represents the merchant/terminal)
        for idx, row in self.data.iterrows():
            cardholder = f"CH_{int(row['Cardholder_ID'])}"
            channel = f"CHAN_{int(row['Channel_ID'])}"
            fraud = int(row['Class'])
            
            # Nodes: represent cardholders and channels (merchants/devices)
            self.graph.add_node(cardholder, type='cardholder')
            self.graph.add_node(channel, type='channel')
            
            # Edges: represent the transaction from cardholder through a channel
            if self.graph.has_edge(cardholder, channel):
                self.graph[cardholder][channel]['weight'] += 1
                self.graph[cardholder][channel]['fraud_total'] += fraud
            else:
                self.graph.add_edge(cardholder, channel, weight=1, fraud_total=fraud)
        
        logger.info(f"Graph Construction Statistics:")
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        logger.info(f"  - Number of Nodes: {num_nodes:,}")
        logger.info(f"  - Number of Edges: {num_edges:,}")
        
        density = nx.density(self.graph)
        logger.info(f"  - Network Density: {density:.6f}")
        
        return self.graph

    def compute_centrality(self):
        """
        Computes various graph metrics for anomaly detection.
        High centrality can often indicate suspicious bridging entities.
        """
        if self.graph is None:
            self.build_network()
            
        logger.info("Computing centrality measures (PageRank, Betweenness)...")
        pr = compute_pagerank(self.graph)
        bt = compute_betweenness_centrality(self.graph)
        dc = nx.degree_centrality(self.graph)
        
        self.centrality_scores = {
            'pagerank': pr,
            'betweenness': bt,
            'degree': dc
        }
        
        logger.info("Centrality computation complete.")
        return self.centrality_scores

    def get_graph_stats(self):
        """
        Log detailed degree distribution and graph properties.
        """
        if self.graph is None:
            return None
            
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        
        logger.info("-" * 40)
        logger.info("GRAPH STATISTICS:")
        logger.info(f"  - Average Degree: {avg_degree:.2f}")
        logger.info(f"  - Maximum Degree: {max_degree}")
        logger.info(f"  - Degrees (Q1/Median/Q3): {np.percentile(degrees, [25, 50, 75])}")
        logger.info("-" * 40)
        
        return {
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges()
        }

    def visualize_topology(self, save_path='06_network_topology.png'):
        """
        Saves a visualization of the most connected part of the network.
        """
        logger.info("Creating network topology visualization (Top 30 nodes)...")
        # Identify top 30 nodes by degree
        top_nodes = sorted([(n, self.graph.degree(n)) for n in self.graph.nodes()],
                           key=lambda x: x[1], reverse=True)[:30]
        top_node_ids = [n for n, d in top_nodes]
        subgraph = self.graph.subgraph(top_node_ids)
        
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        nodes = list(subgraph.nodes())
        node_colors = ['#3498db' if 'CH_' in n else '#2ecc71' for n in nodes]
        
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=300, alpha=0.9, edgecolors='black')
        nx.draw_networkx_edges(subgraph, pos, edge_color='#95a5a6', arrows=True, alpha=0.5)
        nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight='bold')
        
        plt.title('FrauduLens: Top 30 Transaction Entities', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Topology saved to: {save_path}")

    # ============================================================================
    # ADVANCED ANOMALY DETECTION METHODS
    # ============================================================================

    def snowball_expansion(self, start_nodes, expansion_steps=3):
        """
        Grows a subgraph from seeds to identify related entities.
        """
        logger.info(f"Running snowball expansion from {start_nodes} for {expansion_steps} steps...")
        group = set(start_nodes)
        frontier = set(start_nodes)
        for _ in range(expansion_steps):
            next_frontier = set()
            for node in frontier:
                if self.graph.has_node(node):
                    neighbors = list(self.graph.neighbors(node))
                    next_frontier.update(neighbors)
            next_frontier.difference_update(group)
            group.update(next_frontier)
            frontier = next_frontier
        logger.info(f"Snowball expansion finished with group size = {len(group)}")
        return group

    def detect_cliques(self):
        """
        Identifies dense subgraphs (fraud rings).
        """
        logger.info("Detecting cliques...")
        undirected = self.graph.to_undirected()
        cliques = list(nx.find_cliques(undirected))
        significant_cliques = [c for c in cliques if len(c) > 2]
        logger.info(f"Detected {len(significant_cliques)} significant cliques (size > 2)")
        return significant_cliques

    def detect_stars(self, threshold=5):
        """
        Identifies 'Star' patterns where a hub interacts with many entities.
        """
        logger.info(f"Detecting star patterns (hubs with degree > {threshold})...")
        stars = []
        for node in self.graph.nodes():
            degree = max(self.graph.out_degree(node), self.graph.in_degree(node))
            if degree > threshold:
                stars.append((node, degree))
        stars.sort(key=lambda x: x[1], reverse=True)
        return stars

    def detect_cycles(self, max_cycles=10):
        """
        Detects simple cycles which can indicate multi-hop money laundering.
        """
        logger.info(f"Detecting simple cycles (max {max_cycles})...")
        try:
            cycles = list(nx.simple_cycles(self.graph))
            filtered = [c for c in cycles if len(c) > 2][:max_cycles]
            return filtered
        except Exception:
            return []

    def entity_resolution(self):
        """
        Uses connected components to find entities using multiple identifiers.
        """
        logger.info("Performing entity resolution via connected components...")
        components = list(nx.weakly_connected_components(self.graph))
        large_comp = [c for c in components if len(c) > 10]
        logger.info(f"Found {len(large_comp)} large connected entity components")
        return large_comp

    def peer_group_anomaly_detection(self):
        """
        Baseline peer-group anomaly detection based on statistical deviance.
        """
        logger.info("Performing peer group anomaly detection...")
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
        
        self.peer_features = feature_df.sort_values('anomaly_score', ascending=False)
        return self.peer_features

    # Dash compatibility aliases
    def clique_detection(self): return self.detect_cliques()
    def compute_centrality_and_plot(self): return self.compute_centrality()
