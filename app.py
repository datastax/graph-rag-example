"""
This module sets up a Dash web application for comparing results from different search methods
(vector similarity, graph traversal, and Maximal Marginal Relevance (MMR)).

The application includes:
- A text input field for users to enter a question.
- A button to trigger the search and retrieve results.
- Sections to display the results, elapsed time, and usage metadata for each search method.

The main components of the module are:
- Initialization of the Dash app and its layout.
- Callback functions to handle the search requests and update the results on the web page.

The application is run in debug mode when executed as the main module.
"""
import logging
import time
import asyncio
import coloredlogs
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_loading_spinners as dls
from search_executor import (
    ChainManager,
    get_similarity_result,
    get_mmr_result
)

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

# ASCII art to be logged at the start of the app
ASCII_ART = """
  ____                 _     ____      _    ____ 
 / ___|_ __ __ _ _ __ | |__ |  _ \    / \  / ___|
| |  _| '__/ _` | '_ \| '_ \| |_) |  / _ \| |  _ 
| |_| | | | (_| | |_) | | | |  _ <  / ___ \ |_| |
 \____|_|  \__,_| .__/|_| |_|_| \_\/_/   \_\____|
                |_|                                           
                        *no graph database needed!!!
"""
logger.info(ASCII_ART)

# Initialize the Dash app
external_stylesheets = ['/assets/globals.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Similarity vs Traversal vs MMR Comparison"),

    html.Div([
        dcc.Input(
            id="question-input",
            type="text",
            value="",
            placeholder="Ask me something",
            className="input-field"
        ),
        html.Button("Get Results", id="submit-button", n_clicks=0, className="button"),
    ], className="input-container"),

    html.Div([
        html.Div([
            html.H2("Vector Similarity Result", className="result-title"),
            dls.Hash(id="similarity-spinner", children=[
                dcc.Markdown(id="similarity-result", className="result-content"),
                html.Div(id="similarity-time", className="result-time"),
                html.Div(id="similarity-usage-metadata", className="usage-metadata")
            ], color="#4CAF50", speed_multiplier=1.5)
        ], className="result-container"),

        html.Div([
            html.H2("MMR Result", className="result-title"),
            dls.Hash(id="mmr-spinner", children=[
                dcc.Markdown(id="mmr-result", className="result-content"),
                html.Div(id="mmr-time", className="result-time"),
                html.Div(id="mmr-usage-metadata", className="usage-metadata")
            ], color="#4CAF50", speed_multiplier=1.5)
        ], className="result-container")
    ], className="results-section")
])


@app.callback(
    [Output("similarity-result", "children"),
     Output("similarity-time", "children"),
     Output("similarity-usage-metadata", "children")],
    [Input("submit-button", "n_clicks")],
    [State("question-input", "value")]
)
def update_similarity_result(n_clicks, question):
    """
    Fetches and updates the similarity result, elapsed time, and usage metadata.
        - n_clicks: Number of times the "Get Results" button has been clicked.
        - question: The input question provided by the user.
    """
    if n_clicks > 0:
        async def fetch_similarity_result(chain_manager, question):
            start_time = time.time()
            result, usage_metadata = await get_similarity_result(chain_manager, question)
            elapsed_time = time.time() - start_time
            return result, usage_metadata, elapsed_time

        chain_manager = ChainManager()
        chain_manager.setup_chains()
        similarity_result, similarity_usage_metadata, similarity_elapsed_time = asyncio.run(
            fetch_similarity_result(chain_manager, question)
        )

        similarity_time = (
            f"Elapsed time: {similarity_elapsed_time:.2f} seconds "
            f"over {len(similarity_result)} documents"
        )
        similarity_usage_metadata_str = f"Usage Metadata: {similarity_usage_metadata}"

        return similarity_result, similarity_time, similarity_usage_metadata_str
    return "", "", ""


@app.callback(
    [Output("mmr-result", "children"),
     Output("mmr-time", "children"),
     Output("mmr-usage-metadata", "children")],
    [Input("submit-button", "n_clicks")],
    [State("question-input", "value")]
)
def update_mmr_result(n_clicks, question):
    """
    Fetches and updates the MMR result, elapsed time, and usage metadata.
        - n_clicks: Number of times the "Get Results" button has been clicked.
        - question: The input question provided by the user.
    """
    if n_clicks > 0:
        async def fetch_mmr_result(chain_manager, question):
            start_time = time.time()
            result, usage_metadata = await get_mmr_result(chain_manager, question)
            elapsed_time = time.time() - start_time
            return result, usage_metadata, elapsed_time

        chain_manager = ChainManager()
        chain_manager.setup_chains()
        mmr_result, mmr_usage_metadata, mmr_elapsed_time = asyncio.run(
            fetch_mmr_result(chain_manager, question)
        )

        mmr_time = (
            f"Elapsed time: {mmr_elapsed_time:.2f} seconds "
            f"over {len(mmr_result)} documents"
        )
        mmr_usage_metadata_str = f"Usage Metadata: {mmr_usage_metadata}"

        return mmr_result, mmr_time, mmr_usage_metadata_str
    return "", "", ""

if __name__ == "__main__":
    app.run_server(debug=True)
