"""
This module provides utility functions for extracting and 
logging links from documents using various link extractors and transformers 
from the langchain_community.graph_vectorstores package.
"""
import re
from typing import Dict, Iterable, Optional
from tabulate import tabulate
from langchain_core.documents import Document
from langchain_core.graph_vectorstores.links import get_links

import graphviz


def _escape_id(id: str) -> str:
    return id.replace(":", "_")

_EDGE_DIRECTION = {
    "in": "back",
    "out": "forward",
    "bidir": "both",
}

_WORD_RE = re.compile("\s*\S+")

def _split_prefix(s: str, max_chars: int = 50) -> str:
    words = _WORD_RE.finditer(s)

    split = min(len(s), max_chars)
    for word in words:
        if word.end(0) > max_chars:
            break
        split = word.end(0)

    if split == len(s):
        return s
    else:
        return f"{s[0:split]}..."


def render_graphviz(
    documents: Iterable[Document],
    engine: Optional[str] = None,
    node_color: Optional[str] = None,
    node_colors: Optional[Dict[str, Optional[str]]] = {},
) -> "graphviz.Digraph":
    """Render a collection of GraphVectorStore documents to GraphViz format.
    Args:
        documents: The documents to render.
        engine: GraphViz layout engine to use. `None` uses the default.
        node_color: General node color. Defaults to `white`.
        node_colors: Dictionary specifying colors of specific nodes. Useful for
            emphasizing nodes that were selected by MMR, or differ from other
            results.
    Returns:
        The "graphviz.Digraph" representing the nodes. May be printed to source,
        or rendered using `dot`.
    Note:
        To render the generated DOT source code, you also need to install Graphviz_
        (`download page <https://www.graphviz.org/download/>`_,
        `archived versions <https://www2.graphviz.org/Archive/stable/>`_,
        `installation procedure for Windows <https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224>`_).
    """
    if node_colors is None:
        node_colors = {}

    try:
        graphviz.version()
    except graphviz.ExecutableNotFound:
        raise ImportError(
            "Could not execute `dot`. "
            "Make sure graphviz executable is installed (see https://www.graphviz.org/download/)."
        )

    tags = set()

    graph = graphviz.Digraph(engine=engine)
    graph.attr(rankdir="LR")
    graph.attr("node", style="filled")
    for document in documents:
        id = document.id
        if id is None:
            raise ValueError(f"Illegal graph document without ID: {document}")
        escaped_id = _escape_id(id)
        color = node_colors[id] if id in node_colors else node_color

        node_label = "\n".join([
            graphviz.escape(id),
            graphviz.escape(_split_prefix(document.page_content)),
        ])
        graph.node(
            escaped_id,
            label=node_label,
            shape="note",
            fillcolor=color,
            tooltip=graphviz.escape(document.page_content),
        )

        for link in get_links(document):
            tag = f"{link.kind}_{link.tag}"
            if tag not in tags:
                graph.node(tag, label=graphviz.escape(f"{link.kind}:{link.tag}"))
                tags.add(tag)

            graph.edge(escaped_id, tag, dir=_EDGE_DIRECTION[link.direction])
    return graph


def visualize_graph(results):
    print("\n\nVisualizing Graph...")
    print("Results: ", results)
    #depth_ids = { d.id for d in results }
    #graph_ids = { d.id for d in results }


    colors = {
        # Depth0, Graph
        # Sky Blue: Both
        (True, True): "#c1e7ff",
        # Blue: Vector-only (displaced by graph results)
        (True, False): "#6996b3",
        # Dark Blue: Graph-only.
        (False, True): "#004c6d",
        (False, False): None,
    }

    #all_documents = results + [d for d in results if d.id not in graph_ids]
    #colors = {
    #    d.id: colors[(d.id in depth_ids, d.id in graph_ids)] for d in all_documents
    #}

    # engine="dot", engine="neato", engine="sfdp"
    render_graphviz(results, engine="fdp", node_colors = colors)


# Function to find and log links
def find_and_log_links(documents):
    """
    Finds and logs cross-document links and collects all hyperlinks from the given documents.
        - documents: List of documents to process.
    """
    links_table = []
    all_hyperlinks = []

    for doc_to in documents:
        for link_to in doc_to.metadata["links"]:
            if link_to.direction == "in":
                for doc_from in documents:
                    for link_from in doc_from.metadata["links"]:
                        if (
                            link_to.direction == "in"
                            and link_from.direction == "out"
                            and link_to.tag == link_from.tag
                        ):
                            links_table.append(
                                [doc_from.metadata['source'],
                                 doc_to.metadata['source']]
                            )
            # Collect all hyperlinks
            if link_to.kind == "hyperlink":
                all_hyperlinks.append(link_to.tag)

    # Format the table for cross-document links
    table_headers = ["Source Document", "Target Document"]
    formatted_table = tabulate(links_table, headers=table_headers, tablefmt="grid")

    # Log the table for cross-document links
    print("\n\nFound Links:\n%s", formatted_table)
