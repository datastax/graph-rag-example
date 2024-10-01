"""
This script initializes and runs a Dash web application for comparing similarity and MMR results.
It includes functions for fetching results asynchronously and updating the UI with the results.
"""
import time
import asyncio
import warnings
import dash
import dash_bootstrap_components as dbc
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
    #visualize_graphs
)
from util.config import LOGGER, DEBUG_MODE

# Suppress all of the Langchain beta and other warnings
warnings.filterwarnings("ignore", lineno=0)

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
LOGGER.info(ASCII_ART)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Similarity vs MMR Comparison"), className="mb-4 text-center")
    ]),

    # User input and get results button
    dbc.Row([
        dbc.Col([
            dbc.Input(
                id="question-input",
                type="text",
                value="",
                placeholder="Ask me something",
                className="input-field mb-2"
            )
        ], width=10),
        dbc.Col([
            dbc.Button("Get Results", id="submit-button", n_clicks=0, color="primary", className="button mb-2")
        ], width=2)
    ], className="justify-content-center mb-4"),

    dbc.Row([
        dbc.Col([  # Normal RAG Section
            dbc.Card([
                dbc.CardHeader(html.H2("Normal RAG", className="result-title")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Num Results:", className="input-label", style={"textAlign": "left", "width": "30%"}),
                            dbc.Label("Number of documents to return", className="input-label", style={"textAlign": "right", "width": "70%"}),
                            dbc.Input(
                                id="k-input-normal",
                                type="number",
                                value=4,
                                placeholder="num_results",
                                className="input-field small-input mb-2"
                            )
                        ])
                    ], align="center"),
                    dls.Hash(id="similarity-spinner", children=[
                        dcc.Markdown(id="similarity-result", className="result-content"),
                        html.Div(id="similarity-time", className="result-time"),
                        html.Div(id="similarity-usage-metadata", className="usage-metadata")
                    ], color="#4CAF50")
                ])
            ], className="result-container")
        ], width=6),

        dbc.Col([  # Graph RAG Section
            dbc.Card([
                dbc.CardHeader(html.H2("Graph RAG", className="result-title")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Num Results", className="input-label"),
                            dbc.Input(
                                id="k-input-graph",
                                type="number",
                                value=4,
                                placeholder="num_results",
                                className="input-field small-input mb-2"
                            )
                        ])
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Depth", className="input-label"),
                            dbc.Input(
                                id="depth-input-graph",
                                type="number",
                                value=2,
                                placeholder="depth",
                                className="input-field small-input mb-2"
                            )
                        ])
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Diverse", className="input-label", style={"textAlign": "left"})
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Relevant", className="input-label", style={"textAlign": "right", "width": "100%"})
                                ], width=6, className="d-flex justify-content-end")
                            ]),
                            dcc.Slider(
                                id="lambda-slider",
                                min=0,
                                max=1,
                                step=0.01,
                                value=0.25,
                                marks={i / 10: str(i / 10) for i in range(0, 11)},
                                className="slider"
                            ),
                        ], width=12),
                    ]),

                    dls.Hash(id="mmr-spinner", children=[
                        dcc.Markdown(id="mmr-result", className="result-content"),
                        html.Div(id="mmr-time", className="result-time"),
                        html.Div(id="mmr-usage-metadata", className="usage-metadata")
                    ], color="#4CAF50")
                ])
            ], className="result-container")
        ], width=6)
    ])
], fluid=True)


async def fetch_similarity_result(chain_manager, question):
    """
    Fetches the similarity result for a given question using the ChainManager.

    Parameters:
    chain_manager (ChainManager): The ChainManager instance to use for fetching the result.
    question (str): The question to fetch the similarity result for.

    Returns:
    tuple: A tuple containing the result, usage metadata, and elapsed time.
    """
    start_time = time.time()
    result, usage_metadata = await get_similarity_result(chain_manager, question)
    elapsed_time = time.time() - start_time
    return result, usage_metadata, elapsed_time


async def fetch_mmr_result(chain_manager, question):
    """
    Fetches the MMR result for a given question using the ChainManager.

    Parameters:
    chain_manager (ChainManager): The ChainManager instance to use for fetching the result.
    question (str): The question to fetch the MMR result for.

    Returns:
    tuple: A tuple containing the result, usage metadata, and elapsed time.
    """
    start_time = time.time()
    result, usage_metadata = await get_mmr_result(chain_manager, question)
    elapsed_time = time.time() - start_time
    return result, usage_metadata, elapsed_time


@app.callback(
    [Output("similarity-result", "children"),
     Output("similarity-time", "children"),
     Output("similarity-usage-metadata", "children")],
    [Input("submit-button", "n_clicks")],
    [State("question-input", "value"),
     State("k-input-normal", "value")]
)
def update_similarity_results(n_clicks, question, k):
    """
    Updates the similarity results in the UI when the submit button is clicked.

    Parameters:
    n_clicks (int): The number of times the submit button has been clicked.
    question (str): The question input by the user.
    k (int): The number of top results to retrieve.
    depth (int): The depth of the graph traversal.

    Returns:
    tuple: A tuple containing the similarity result, elapsed time, and usage metadata.
    """
    if n_clicks > 0:
        chain_manager = ChainManager()
        chain_manager.setup_chains(k=k, depth=0)
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
    [State("question-input", "value"),
     State("k-input-graph", "value"),
     State("depth-input-graph", "value"),
     State("lambda-slider", "value")]
)
def update_mmr_results(n_clicks, question, k, depth, lambda_mult):
    """
    Updates the MMR results in the UI when the submit button is clicked.

    Parameters:
    n_clicks (int): The number of times the submit button has been clicked.
    question (str): The question input by the user.
    k (int): The number of top results to retrieve.
    depth (int): The depth of the graph traversal.
    lambda_mult (float): The lambda multiplier for MMR.

    Returns:
    tuple: A tuple containing the MMR result, elapsed time, and usage metadata.
    """
    if n_clicks > 0:
        chain_manager = ChainManager()
        chain_manager.setup_chains(k=k, depth=depth, lambda_mult=lambda_mult)
        mmr_result, mmr_usage_metadata, mmr_elapsed_time = asyncio.run(
            fetch_mmr_result(chain_manager, question)
        )

        visualize_result = chain_manager.mmr_retriever.invoke(question)
        for result in visualize_result:
            print(f"\n\n {result.metadata.get('source')}")
            print(f"\n\n {result.metadata}")

        if DEBUG_MODE:
            #visualize_graphs(visualize_result)
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
