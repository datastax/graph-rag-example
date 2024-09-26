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
from util.visualization import (
    visualize_graph_text,
    visualize_graphs
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
    html.H1("Similarity vs MMR Comparison"),

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


async def fetch_similarity_result(chain_manager, question):
    start_time = time.time()
    result, usage_metadata = await get_similarity_result(chain_manager, question)
    elapsed_time = time.time() - start_time
    return result, usage_metadata, elapsed_time


async def fetch_mmr_result(chain_manager, question):
    start_time = time.time()
    result, usage_metadata = await get_mmr_result(chain_manager, question)
    elapsed_time = time.time() - start_time
    return result, usage_metadata, elapsed_time


@app.callback(
    [Output("similarity-result", "children"),
     Output("similarity-time", "children"),
     Output("similarity-usage-metadata", "children")],
    [Input("submit-button", "n_clicks")],
    [State("question-input", "value")]
)
def update_similarity_results(n_clicks, question):
    if n_clicks > 0:
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

        return (similarity_result, similarity_time, similarity_usage_metadata_str)
    return "", "", ""


@app.callback(
    [Output("mmr-result", "children"),
     Output("mmr-time", "children"),
     Output("mmr-usage-metadata", "children")],
    [Input("submit-button", "n_clicks")],
    [State("question-input", "value")]
)
def update_mmr_results(n_clicks, question):
    if n_clicks > 0:
        chain_manager = ChainManager()
        chain_manager.setup_chains()
        mmr_result, mmr_usage_metadata, mmr_elapsed_time = asyncio.run(
            fetch_mmr_result(chain_manager, question)
        )

        visualize_result = chain_manager.mmr_retriever.invoke(question)
        visualize_graphs(visualize_result)
        visualize_graph_text(visualize_result, direction="bidir")

        mmr_time = (
            f"Elapsed time: {mmr_elapsed_time:.2f} seconds "
            f"over {len(mmr_result)} documents"
        )
        mmr_usage_metadata_str = f"Usage Metadata: {mmr_usage_metadata}"

        return (mmr_result, mmr_time, mmr_usage_metadata_str)
    return "", "", ""


if __name__ == "__main__":
    app.run_server(debug=True)
