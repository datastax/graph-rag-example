import asyncio
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_loading_spinners as dls
from main import get_similarity_result, get_traversal_result, ChainManager

# Dash front-end implementation
app = dash.Dash(__name__)

app.layout = html.Div([
    # Main title
    html.H1("Model Comparison", style={
        "textAlign": "center", 
        "color": "white", 
        "fontFamily": "Arial", 
        "padding": "10px"}),

    html.Div([
        dcc.Input(id="question-input", type="text", value="What are the latest upcoming movies, their release dates, and URLs?",
                  style={"width": "80%", "padding": "10px", "borderRadius": "10px", "marginRight": "10px"}),
        html.Button("Get Results", id="submit-button", n_clicks=0, style={
            "padding": "10px 20px", 
            "backgroundColor": "#4CAF50", 
            "color": "white", 
            "border": "none", 
            "borderRadius": "5px",
            "cursor": "pointer",
            "transition": "background-color 0.3s ease"}),
    ], style={"textAlign": "center", "padding": "20px"}),

    # Results section
    html.Div([
        html.Div([
            html.H2("Vector Similarity Result", style={"color": "white"}),
            dls.Hash(id="similarity-spinner", children=[
                html.Pre(id="similarity-result", style={"whiteSpace": "pre-wrap", "backgroundColor": "#333", "color": "white", "padding": "20px", "borderRadius": "10px"})
            ], color="#4CAF50", speed_multiplier=1.5)
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "marginRight": "2%", "backgroundColor": "#1E1E1E", "padding": "20px", "borderRadius": "10px"}),

        html.Div([
            html.H2("Graph Traversal Result", style={"color": "white"}),
            dls.Hash(id="traversal-spinner", children=[
                html.Pre(id="traversal-result", style={"whiteSpace": "pre-wrap", "backgroundColor": "#333", "color": "white", "padding": "20px", "borderRadius": "10px"})
            ], color="#4CAF50", speed_multiplier=1.5)
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "backgroundColor": "#1E1E1E", "padding": "20px", "borderRadius": "10px"})
    ], style={"display": "flex", "justifyContent": "center", "padding": "20px"})
], style={"backgroundColor": "#121212", "minHeight": "100vh", "padding": "20px"})


@app.callback(
    [Output("similarity-result", "children"),
     Output("traversal-result", "children"),
     Output("submit-button", "style")],
    [Input("submit-button", "n_clicks")],
    [State("question-input", "value")]
)
def update_output(n_clicks, question):
    if n_clicks > 0:
        async def fetch_results():
            chain_manager = ChainManager()
            chain_manager.setup_chains()
            similarity_result, traversal_result = await asyncio.gather(
                get_similarity_result(chain_manager, question),
                get_traversal_result(chain_manager, question)
            )
            return similarity_result, traversal_result

        similarity_result, traversal_result = asyncio.run(fetch_results())

        button_style = {
            "padding": "10px 20px", 
            "backgroundColor": "#3e8e41", 
            "color": "white", 
            "border": "none", 
            "borderRadius": "5px",
            "cursor": "pointer",
            "transition": "background-color 0.3s ease"
        }
        return similarity_result, traversal_result, button_style
    return "", "", {
        "padding": "10px 20px", 
        "backgroundColor": "#4CAF50", 
        "color": "white", 
        "border": "none", 
        "borderRadius": "5px",
        "cursor": "pointer",
        "transition": "background-color 0.3s ease"
    }

if __name__ == "__main__":
    app.run_server(debug=True)
