from tabulate import tabulate
from langchain_community.graph_vectorstores.extractors import (
    LinkExtractorTransformer,
    HtmlLinkExtractor,
    KeybertLinkExtractor,
)
from langchain_community.graph_vectorstores.links import add_links

# Function to find and log links
def find_and_log_links(documents):
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


def use_as_document_extractor(documents):
    html_extractor = HtmlLinkExtractor().as_document_extractor()

    for document in documents:
        links = html_extractor.extract_one(document)
        add_links(document, links)
    return documents


def use_link_extractor_transformer(documents):
    transformer = LinkExtractorTransformer([HtmlLinkExtractor().as_document_extractor()])
    documents = transformer.transform_documents(documents)
    return documents


def use_keybert_extract_one(documents):
    keyword_extractor = KeybertLinkExtractor()

    for document in documents:
        links = keyword_extractor.extract_one(document)
        add_links(document, links)
    return documents


def use_keybert_extractor(documents):
    transformer = LinkExtractorTransformer([KeybertLinkExtractor()])
    documents = transformer.transform_documents(documents)
    return documents
