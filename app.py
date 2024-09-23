import asyncio
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_loading_spinners as dls
from main import get_similarity_result, get_traversal_result, ChainManager

# Dash front-end implementation
external_stylesheets = ['/assets/globals.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    # Main title
    html.H1("Similarity vs Traversal Comparison"),

    html.Div([
        dcc.Input(id="question-input", type="text", value="",
                  className="input-field"),
        html.Button("Get Results", id="submit-button", n_clicks=0, className="button"),
    ], className="input-container"),

    # Results section
    html.Div([
        html.Div([
            html.H2("Vector Similarity Result", className="result-title"),
            dls.Hash(id="similarity-spinner", children=[
                html.Pre(id="similarity-result", className="result-content"),
                html.Div(id="similarity-time", className="result-time")
            ], color="#4CAF50", speed_multiplier=1.5)
        ], className="result-container"),

        html.Div([
            html.H2("Graph Traversal Result", className="result-title"),
            dls.Hash(id="traversal-spinner", children=[
                html.Pre(id="traversal-result", className="result-content"),
                html.Div(id="traversal-time", className="result-time")
            ], color="#4CAF50", speed_multiplier=1.5)
        ], className="result-container")
    ], className="results-section")
])

@app.callback(
    [Output("similarity-result", "children"),
     Output("traversal-result", "children"),
     Output("similarity-time", "children"),
     Output("traversal-time", "children"),
     Output("submit-button", "className")],
    [Input("submit-button", "n_clicks")],
    [State("question-input", "value")]
)
def update_output(n_clicks, question):
    if n_clicks > 0:
        # Clear the result fields to reset their size
        similarity_result = ""
        traversal_result = ""
        similarity_time = ""
        traversal_time = ""

        async def fetch_results():
            chain_manager = ChainManager()
            chain_manager.setup_chains()
            start_time = time.time()
            similarity_result, traversal_result = await asyncio.gather(
                get_similarity_result(chain_manager, question),
                get_traversal_result(chain_manager, question)
            )
            elapsed_time = time.time() - start_time
            return similarity_result, traversal_result, elapsed_time

        similarity_result, traversal_result, elapsed_time = asyncio.run(fetch_results())

        button_class = "button button-clicked"
        similarity_time = f"Elapsed time: {elapsed_time:.2f} seconds"
        traversal_time = f"Elapsed time: {elapsed_time:.2f} seconds"
        return similarity_result, traversal_result, similarity_time, traversal_time, button_class
    return "", "", "", "", "button"

if __name__ == "__main__":
    app.run_server(debug=True)
