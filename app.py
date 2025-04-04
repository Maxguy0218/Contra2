import dash
from dash import dcc, html, Input, Output, State, dash_table, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pdfplumber
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import base64
import io
import os

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "FedEx ContractIQ"
server = app.server  # Required for deployment

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Color Scheme
colors = {
    "background": "#FFFFFF",
    "text": "#4D148C",
    "primary": "#FF6200",
    "secondary": "#4D148C"
}

# Layout Components
sidebar = html.Div([
    dbc.Button("⚙️ Config", id="open-sidebar", className="mb-2"),
    dbc.Collapse(
        dbc.Card([
            dbc.CardHeader("Configuration"),
            dbc.CardBody([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag & Drop or ', html.A('Select PDFs')]),
                    multiple=True,
                    style={
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'padding': '10px'
                    }
                ),
                dbc.Label("AI Model"),
                dbc.Select(
                    id='ai-model',
                    options=[
                        {"label": "Transportation & Logistics", "value": "logistics"},
                        {"label": "Warehousing & Storage", "value": "warehouse"}
                    ],
                    value="logistics"
                )
            ])
        ]),
        id="sidebar-collapse",
        is_open=False
    )
], style={'position': 'fixed', 'left': '0', 'padding': '20px'})

content = html.Div([
    dbc.Tabs([
        dbc.Tab(label="Critical Data", tab_id="critical", children=[
            dash_table.DataTable(
                id='critical-table',
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': colors['secondary'],
                    'color': 'white',
                    'fontWeight': 'bold'
                }
            ),
            dcc.Graph(id='donut-chart')
        ]),
        dbc.Tab(label="Commercial", tab_id="commercial", children=[
            dash_table.DataTable(
                id='commercial-table',
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': colors['secondary'],
                    'color': 'white',
                    'fontWeight': 'bold'
                }
            )
        ]),
        dbc.Tab(label="Legal", tab_id="legal", children=[
            dash_table.DataTable(
                id='legal-table',
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': colors['secondary'],
                    'color': 'white',
                    'fontWeight': 'bold'
                }
            )
        ])
    ], id="tabs", active_tab="critical"),
    
    html.Div([
        dcc.Input(id='chat-input', placeholder="Ask about contracts...", 
                 style={'width': '80%', 'marginRight': '10px'}),
        dbc.Button("Ask", id='ask-button', color="primary")
    ], style={'position': 'fixed', 'bottom': '20px', 'width': '100%', 'padding': '20px'}),
    
    html.Div(id='chat-output', style={
        'position': 'fixed',
        'bottom': '80px',
        'width': '100%',
        'height': '200px',
        'overflowY': 'scroll',
        'padding': '20px'
    }),
    
    dcc.Store(id='processed-data'),
    dcc.Store(id='vector-store')
], style={'marginLeft': '250px'})

app.layout = html.Div([sidebar, content])

# Callbacks
@callback(
    Output("sidebar-collapse", "is_open"),
    Input("open-sidebar", "n_clicks"),
    State("sidebar-collapse", "is_open")
)
def toggle_sidebar(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
    [Output('processed-data', 'data'),
     Output('vector-store', 'data')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def process_files(contents, filenames):
    if not contents:
        return dash.no_update
    
    texts = []
    for content, filename in zip(contents, filenames):
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        with pdfplumber.open(io.BytesIO(decoded)) as pdf:
            texts.append("\n".join([page.extract_text() or "" for page in pdf.pages]))
    
    # Create vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings)
    
    # Mock data - replace with actual data processing
    critical_data = pd.DataFrame(CRITICAL_DATA).to_dict('records')
    commercial_data = pd.DataFrame(COMMERCIAL_DATA).to_dict('records')
    legal_data = pd.DataFrame(LEGAL_DATA).to_dict('records')
    
    return {
        'critical': critical_data,
        'commercial': commercial_data,
        'legal': legal_data
    }, vector_store.to_json()

@callback(
    [Output('critical-table', 'data'),
     Output('commercial-table', 'data'),
     Output('legal-table', 'data'),
     Output('donut-chart', 'figure')],
    Input('processed-data', 'data')
)
def update_tables(data):
    if not data:
        return [], [], [], {}
    
    # Create donut chart
    contract_types = pd.DataFrame(data['critical'])['Type of Contract'].value_counts().reset_index()
    fig = px.pie(contract_types, values='count', names='Type of Contract', hole=0.4)
    fig.update_traces(marker=dict(colors=[colors['primary'], colors['secondary']))
    
    return data['critical'], data['commercial'], data['legal'], fig

@callback(
    Output('chat-output', 'children'),
    Input('ask-button', 'n_clicks'),
    State('chat-input', 'value'),
    State('vector-store', 'data')
)
def handle_ask(n_clicks, question, vector_store_json):
    if not n_clicks or not question:
        return dash.no_update
    
    vector_store = FAISS.load_local(vector_store_json, embeddings=HuggingFaceEmbeddings())
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    response = model.generate_content(f"Context:\n{context}\n\nQuestion: {question}")
    return html.Div([
        html.Div(f"You: {question}", style={'color': colors['secondary']}),
        html.Div(f"AI: {response.text}", style={'color': colors['primary'], 'marginTop': '10px'})
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
