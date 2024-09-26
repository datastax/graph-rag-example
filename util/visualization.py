import re
from typing import TYPE_CHECKING, Iterable, Optional, Dict, Tuple
from tabulate import tabulate
from anytree import Node, RenderTree, LoopError
from langchain_core.documents import Document
from langchain_community.graph_vectorstores.links import get_links

if TYPE_CHECKING:
    import graphviz

_EDGE_DIRECTION = {
    "in": "back",
    "out": "forward",
    "bidir": "both",
}

_WORD_RE = re.compile("\s*\S+")

COLORS = {
    True: "green",
    False: "red",
    # Depth0, Graph
    # Sky Blue: Both
    (True, True): "#c1e7ff",
    # Blue: Left-Only
    (True, False): "#6996b3",
    # Dark Blue: Right-Only
    (False, True): "#004c6d",
    (False, False): None,
}


def _escape_id(id: str) -> str:
    return id.replace(":", "_")


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
    node_colors: Optional[Dict[str, Optional[str]]] = None,
    skip_tags: Iterable[Tuple[str, str]] = (),
) -> "graphviz.Digraph":
    """Render a collection of GraphVectorStore documents to GraphViz format.
    Args:
        documents: The documents to render.
        engine: GraphViz layout engine to use. `None` uses the default.
        node_color: Default node color.
        node_colors: Dictionary specifying colors of specific nodes. Useful for
            emphasizing nodes that were selected by MMR, or differ from other
            results.
        skip_tags: Set of tags to skip when rendering the graph. Specified as
            tuples containing the kind and tag.
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
        import graphviz
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Could not import graphviz python package. "
            "Please install it with `pip install graphviz`."
        )

    graph = graphviz.Digraph(engine=engine)
    graph.attr(rankdir="LR")
    graph.attr("node", style="filled")

    skip_tags = set(skip_tags)
    tags: dict[Tuple[str, str], str] = {}

    for document in documents:
        id = document.id
        if id is None:
            raise ValueError(f"Illegal graph document without ID: {document}")
        escaped_id = _escape_id(id)
        color = node_colors[id] if id in node_colors else node_color

        node_label = "\n".join(
            [
                graphviz.escape(id),
                graphviz.escape(_split_prefix(document.page_content)),
            ]
        )
        graph.node(
            escaped_id,
            label=node_label,
            shape="note",
            fillcolor=color,
            tooltip=graphviz.escape(document.page_content),
        )

        for link in get_links(document):
            tag_key = (link.kind, link.tag)
            if tag_key in skip_tags:
                continue

            tag_id = tags.get(tag_key)
            if tag_id is None:
                tag_id = f"tag_{len(tags)}"
                tags[tag_key] = tag_id
                graph.node(tag_id, label=graphviz.escape(f"{link.kind}:{link.tag}"))

            graph.edge(escaped_id, tag_id, dir=_EDGE_DIRECTION[link.direction])
    return graph


def visualize_graphs(documents, output_path="graph"):
    document_ids = { d.id for d in documents }
    #left_ids = { d.id for d in left_results }
    #right_ids = { d.id for d in right_results }

    colors = {
      d.id: COLORS[(d.id in document_ids)] for d in documents
    }
    #all_documents = left_results + [d for d in right_results if d.id not in left_ids]
    #colors = {
    #  d.id: COLORS[(d.id in left_ids, d.id in right_ids)] for d in all_documents
    #}

    digraph = render_graphviz(documents, engine="sfdp", node_colors=colors)
    return digraph.render(output_path, format="png")



def generate_links_table(documents, direction="bidir"):
    """
    Generates a table of links between documents.
    
    Parameters:
    documents (list): List of documents to process.
    direction (str): Direction of the links to include ("bidir", "in", "out").
    
    Returns:
    list: List of tuples representing the links.
    """
    links_table = []
    all_links = set()

    # Collect all links
    for doc in documents:
        source = doc.metadata.get("source")
        for link in doc.metadata.get("links", []):
            all_links.add((source, link.tag, link.direction))

    # Filter links based on direction
    for source, tag, link_direction in all_links:
        #print(f"\nSource: {source}, Tag: {tag}, Link Direction: {link_direction}")
        if direction == "bidir" and link_direction == "bidir":
            #print(f"Checking for bidirectional links between {source} and {tag}")
            links_table.append((source, tag))
        elif direction == "out" and link_direction == "out":
            #print(f"Checking for outgoing links from {source} to {tag}")
            links_table.append((source, tag))
        elif direction == "in" and link_direction == "in":
            #print(f"Checking for incoming links to {source} from {tag}")
            links_table.append((tag, source))

    return links_table


def visualize_graph_text(documents, direction="bidir") -> RenderTree:
    print("\n\nVisualizing Text Graph...")

    # Use the updated generate_links_table function
    links_table = generate_links_table(documents, direction)

    # Use a dictionary to hold nodes
    nodes = {}

    # Establish parent-child relationships
    for link in links_table:
        doc_from = link[0]
        doc_to = link[1]
        # Ensure both nodes exist in the dictionary
        if doc_from not in nodes:
            nodes[doc_from] = Node(doc_from)
        if (doc_to not in nodes):
            nodes[doc_to] = Node(doc_to)
        # Check for loops before setting the parent
        try:
            nodes[doc_to].parent = nodes[doc_from]
            #print(f"{doc_from} is parent of {doc_to}")
        except LoopError as e:
            print(f"Skipping loop creation: {e}")

    # Identify all root nodes (nodes without parents)
    root_nodes = [node for node in nodes.values() if node.is_root]

    # Collect the rendered tree structures
    rendered_trees = []
    print("\nTree Structure:")
    for root_node in root_nodes:
        tree_str = ""
        for pre, fill, node in RenderTree(root_node):
            tree_str += "%s%s\n" % (pre, node.name)
        rendered_trees.append(tree_str)
        print(tree_str)

    # Combine all rendered trees into a single string
    combined_tree_str = "".join(rendered_trees)

    return combined_tree_str
