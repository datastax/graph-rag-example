from tabulate import tabulate
from anytree import Node, RenderTree, LoopError


def generate_links_table(documents):
    # This function should generate a table of links between documents
    # For simplicity, let's assume it returns a list of tuples (doc_from, doc_to)
    links_table = []
    for doc in documents:
        for link in doc.metadata.get("links", []):
            links_table.append((doc.metadata["source"], link))
    return links_table


def visualize_graph_text(documents):
    print("\n\nVisualizing Text Graph...")

    # Use a dictionary to hold nodes
    nodes = {}

    # Establish parent-child relationships
    links_table = generate_links_table(documents)
    for link in links_table:
        doc_from = link[0]
        doc_to = link[1]
        # Ensure both nodes exist in the dictionary
        if doc_from not in nodes:
            nodes[doc_from] = Node(doc_from)
        if doc_to not in nodes:
            nodes[doc_to] = Node(doc_to)
        # Check for loops before setting the parent
        try:
            nodes[doc_to].parent = nodes[doc_from]
            print(f"{doc_from} is parent of {doc_to}")
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


# Function to find and log links
def find_and_log_links(documents):
    links_table = generate_links_table(documents)

    # Format the table for cross-document links
    table_headers = ["Source Document", "Target Document"]
    formatted_table = tabulate(links_table, headers=table_headers, tablefmt="pipe")

    # Log the table for cross-document links
    print("\n\nFound Links:\n%s", formatted_table)
