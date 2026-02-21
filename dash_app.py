import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
from fraud_detection_FINAL import DataPreprocessor, NetworkAnalyzer
from fraud_ml_extensions import run_ml_extension
from plotly_dashboard import PlotlyVisualizer
import os

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Credit Card Fraud Detection Dashboard"

# Global variables to store processed data
processed_data = None
network_analyzer = None
ml_detector = None
plotly_viz = None

def load_data():
    """Load and process data for the dashboard"""
    global processed_data, network_analyzer, ml_detector, plotly_viz
    
    print("[*] Loading data for dashboard...")
    preprocessor = DataPreprocessor()
    
    # Try multiple possible locations for the dataset
    possible_paths = [
        'creditcard.csv',
        './creditcard.csv',
        '../creditcard.csv',
        'data/creditcard.csv',
        './data/creditcard.csv'
    ]
    
    raw_data = None
    dataset_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"[*] Found dataset at: {path}")
            break
    
    if dataset_path is None:
        print("[!] Error: creditcard.csv not found!")
        print("[!] Searched in: " + ", ".join(possible_paths))
        print("[!] Please download the dataset from:")
        print("[!] https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("[!] And place it in the project directory as 'creditcard.csv'")
        return False
    
    try:
        raw_data = preprocessor.load_kaggle_dataset(dataset_path)
    except FileNotFoundError:
        print(f"[!] Error: Could not load dataset from {dataset_path}")
        return False
    
    processed_data = preprocessor.preprocess_data()
    plotly_viz = PlotlyVisualizer(processed_data)
    
    # Build network
    network_analyzer = NetworkAnalyzer(processed_data)
    network_analyzer.build_network()
    network_analyzer.compute_centrality_and_plot()
    network_analyzer.peer_group_anomaly_detection()
    
    # Run ML extension
    print("[*] Running ML models...")
    ml_detector = run_ml_extension(processed_data, network_analyzer)
    
    print("[✓] Data loaded successfully!")
    return True

# Load data on startup
data_loaded = load_data()

# Define app layout
app.layout = html.Div([
    html.Div([
        html.H1("💳 Credit Card Fraud Detection Dashboard", 
               style={'textAlign': 'center', 'marginBottom': 10, 'color': '#2c3e50'}),
        html.P("Interactive Analytics & Machine Learning Insights", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 30})
    ]),
    
    dcc.Tabs(id="tabs", value='overview', children=[
        dcc.Tab(label='📊 Overview', value='overview'),
        dcc.Tab(label='🔍 Network Analysis', value='network'),
        dcc.Tab(label='🎯 Fraud Detection', value='fraud'),
        dcc.Tab(label='📈 ML Insights', value='ml'),
        dcc.Tab(label='📋 Data Explorer', value='data'),
    ], style={'fontSize': 16}),
    
    html.Div(id='tab-content', style={'marginTop': 20})
])

@app.callback(Output('tab-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if not data_loaded or processed_data is None:
        return html.Div([
            html.Div([
                html.H2("⚠️ Dataset Not Found", style={'color': '#e74c3c', 'textAlign': 'center'}),
                html.P("Please ensure 'creditcard.csv' is in the current directory.", 
                       style={'textAlign': 'center', 'fontSize': 18}),
                html.P("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
                       style={'textAlign': 'center', 'fontSize': 16, 'color': '#3498db'}),
            ], style={'padding': '50px', 'maxWidth': '800px', 'margin': '0 auto'})
        ])
    
    if tab == 'overview':
        fraud_count = processed_data['Class'].sum()
        total_count = len(processed_data)
        fraud_rate = (fraud_count / total_count * 100) if total_count > 0 else 0
        total_amount = processed_data['Amount'].sum()
        fraud_amount = processed_data[processed_data['Class'] == 1]['Amount'].sum()
        
        return html.Div([
            html.Div([
                html.Div([
                    html.H3(f"{total_count:,}", style={'color': '#3498db', 'fontSize': 48, 'margin': 0}),
                    html.P("Total Transactions", style={'fontSize': 18, 'margin': '5px 0'})
                ], className='stat-box', style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'padding': '25px',
                    'borderRadius': '10px',
                    'color': 'white',
                    'textAlign': 'center',
                    'minWidth': '200px',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                }),
                html.Div([
                    html.H3(f"{fraud_count:,}", style={'color': '#e74c3c', 'fontSize': 48, 'margin': 0}),
                    html.P("Fraudulent Transactions", style={'fontSize': 18, 'margin': '5px 0'})
                ], className='stat-box', style={
                    'background': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                    'padding': '25px',
                    'borderRadius': '10px',
                    'color': 'white',
                    'textAlign': 'center',
                    'minWidth': '200px',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                }),
                html.Div([
                    html.H3(f"{fraud_rate:.2f}%", style={'color': '#e74c3c', 'fontSize': 48, 'margin': 0}),
                    html.P("Fraud Rate", style={'fontSize': 18, 'margin': '5px 0'})
                ], className='stat-box', style={
                    'background': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
                    'padding': '25px',
                    'borderRadius': '10px',
                    'color': 'white',
                    'textAlign': 'center',
                    'minWidth': '200px',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                }),
                html.Div([
                    html.H3(f"${total_amount:,.2f}", style={'color': '#2ecc71', 'fontSize': 48, 'margin': 0}),
                    html.P("Total Amount", style={'fontSize': 18, 'margin': '5px 0'}),
                    html.P(f"Fraud: ${fraud_amount:,.2f}", style={'fontSize': 14, 'margin': '5px 0', 'opacity': 0.9})
                ], className='stat-box', style={
                    'background': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                    'padding': '25px',
                    'borderRadius': '10px',
                    'color': 'white',
                    'textAlign': 'center',
                    'minWidth': '200px',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                }),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap', 
                     'marginBottom': 30, 'gap': '20px'}),
            
            dcc.Graph(figure=plotly_viz.create_fraud_distribution(), 
                     style={'marginBottom': 30}),
            dcc.Graph(figure=plotly_viz.create_amount_distribution(),
                     style={'marginBottom': 30}),
            dcc.Graph(figure=plotly_viz.create_time_distribution(),
                     style={'marginBottom': 30}),
            dcc.Graph(figure=plotly_viz.create_feature_correlation()),
        ], style={'padding': '20px'})
    
    elif tab == 'network':
        num_nodes = network_analyzer.graph.number_of_nodes()
        num_edges = network_analyzer.graph.number_of_edges()
        
        return html.Div([
            html.Div([
                html.Div([
                    html.H3(f"{num_nodes:,}", style={'color': '#3498db', 'fontSize': 36, 'margin': 0}),
                    html.P("Network Nodes", style={'fontSize': 16, 'margin': '5px 0'})
                ], style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'color': 'white',
                    'textAlign': 'center',
                    'minWidth': '200px',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                }),
                html.Div([
                    html.H3(f"{num_edges:,}", style={'color': '#9b59b6', 'fontSize': 36, 'margin': 0}),
                    html.P("Network Edges", style={'fontSize': 16, 'margin': '5px 0'})
                ], style={
                    'background': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'color': 'white',
                    'textAlign': 'center',
                    'minWidth': '200px',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                }),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap',
                     'marginBottom': 30, 'gap': '20px'}),
            
            dcc.Graph(figure=plotly_viz.create_network_statistics(network_analyzer.graph),
                     style={'marginBottom': 30}),
            dcc.Graph(figure=plotly_viz.create_network_topology(network_analyzer.graph),
                     style={'marginBottom': 30}),
            dcc.Graph(figure=plotly_viz.create_centrality_comparison(network_analyzer.centrality_scores),
                     style={'marginBottom': 30}),
            dcc.Graph(figure=plotly_viz.create_anomaly_scores(network_analyzer.peer_features)),
        ], style={'padding': '20px'})
    
    elif tab == 'fraud':
        return html.Div([
            html.H2("Combined Fraud Risk Analysis", style={'textAlign': 'center', 'marginBottom': 30}),
            dcc.Graph(figure=plotly_viz.create_combined_fraud_heatmap(network_analyzer),
                     style={'marginBottom': 30}),
            dcc.Graph(figure=plotly_viz.create_fraud_bubble_chart(network_analyzer)),
        ], style={'padding': '20px'})
    
    elif tab == 'ml':
        anomaly_count = ml_detector.data['anomaly_flag'].sum() if ml_detector else 0
        
        return html.Div([
            html.H2("Machine Learning Insights", style={'textAlign': 'center', 'marginBottom': 30}),
            
            html.Div([
                html.Div([
                    html.H3("XGBoost", style={'color': '#3498db', 'fontSize': 24}),
                    html.P("Hybrid Model with Graph Features", style={'fontSize': 16}),
                    html.P("SMOTE Balanced Training", style={'fontSize': 14, 'color': '#7f8c8d'}),
                    html.P("Combines tabular + network features", style={'fontSize': 14, 'color': '#7f8c8d'})
                ], style={
                    'background': '#f8f9fa',
                    'padding': '25px',
                    'borderRadius': '10px',
                    'border': '2px solid #3498db',
                    'textAlign': 'center',
                    'minWidth': '250px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
                html.Div([
                    html.H3("Isolation Forest", style={'color': '#e74c3c', 'fontSize': 24}),
                    html.P("Unsupervised Anomaly Detection", style={'fontSize': 16}),
                    html.P(f"Flagged: {anomaly_count:,} anomalies", style={'fontSize': 14, 'color': '#e74c3c', 'fontWeight': 'bold'}),
                    html.P("Zero-day fraud detection", style={'fontSize': 14, 'color': '#7f8c8d'})
                ], style={
                    'background': '#f8f9fa',
                    'padding': '25px',
                    'borderRadius': '10px',
                    'border': '2px solid #e74c3c',
                    'textAlign': 'center',
                    'minWidth': '250px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap',
                     'marginBottom': 30, 'gap': '20px'}),
            
            html.Div([
                html.P("Feature Importance Analysis", style={'fontSize': 18, 'fontWeight': 'bold', 'textAlign': 'center'}),
                html.Img(src='ml_feature_importance.png', 
                        style={'width': '100%', 'maxWidth': '1200px', 'display': 'block', 'margin': '0 auto'},
                        alt='Feature Importance Chart')
            ]) if os.path.exists('ml_feature_importance.png') else html.Div([
                html.P("Run the ML pipeline to generate feature importance chart", 
                      style={'textAlign': 'center', 'color': '#7f8c8d'})
            ]),
        ], style={'padding': '20px'})
    
    elif tab == 'data':
        return html.Div([
            html.H2("Data Explorer", style={'textAlign': 'center', 'marginBottom': 30}),
            
            html.Div([
                html.Label("Filter by Fraud Status:", style={'fontSize': 16, 'fontWeight': 'bold', 'marginRight': 10}),
                dcc.Dropdown(
                    id='fraud-filter',
                    options=[
                        {'label': 'All Transactions', 'value': 'all'},
                        {'label': 'Fraud Only', 'value': 'fraud'},
                        {'label': 'Normal Only', 'value': 'normal'}
                    ],
                    value='all',
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'marginBottom': 20, 'display': 'flex', 'alignItems': 'center'}),
            
            html.Div([
                html.Label("Rows per page:", style={'fontSize': 16, 'fontWeight': 'bold', 'marginRight': 10}),
                dcc.Dropdown(
                    id='page-size',
                    options=[{'label': str(i), 'value': i} for i in [10, 25, 50, 100]],
                    value=25,
                    style={'width': '100px', 'display': 'inline-block'}
                ),
            ], style={'marginBottom': 20, 'display': 'flex', 'alignItems': 'center'}),
            
            dash_table.DataTable(
                id='data-table',
                columns=[{"name": i, "id": i} for i in processed_data.columns],
                data=processed_data.head(100).to_dict('records'),
                page_size=25,
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'fontFamily': 'Arial',
                    'fontSize': '12px'
                },
                style_header={
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Class} = 1'},
                        'backgroundColor': '#ffebee',
                        'color': '#c62828',
                    }
                ],
                filter_action="native",
                sort_action="native",
                page_action="native",
            ),
        ], style={'padding': '20px'})

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }
            .dash-tabs {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starting Credit Card Fraud Detection Dashboard")
    print("="*80)
    print("\n📊 Dashboard will be available at: http://localhost:8050")
    print("💡 Press Ctrl+C to stop the server\n")
    app.run(debug=True, port=8050)
