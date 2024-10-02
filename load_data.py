"""
This script loads, processes, and visualizes documents from a list of URLs.
It includes functions for fetching URLs, cleaning and preprocessing documents,
and adding them to a graph vector store.
"""
import os
import json
import warnings
import cassio

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from langchain_community.graph_vectorstores.extractors import (
    LinkExtractorTransformer,
    HtmlLinkExtractor,
    KeybertLinkExtractor,
    GLiNERLinkExtractor,
)
from langchain_community.document_transformers import BeautifulSoupTransformer

from util.config import LOGGER, OPENAI_API_KEY, ASTRA_DB_ID, ASTRA_TOKEN, MOVIE_NODE_TABLE
from util.scrub import clean_and_preprocess_documents
from util.visualization import visualize_graph_text

# Suppress all of the Langchain beta and other warnings
warnings.filterwarnings("ignore", lineno=0)

# Initialize embeddings and LLM using OpenAI
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Initialize Astra connection using Cassio
cassio.init(database_id=ASTRA_DB_ID, token=ASTRA_TOKEN)
store = CassandraGraphVectorStore(embeddings, node_table=MOVIE_NODE_TABLE)


def get_urls(num_items=10):
    """
    Fetches a list of URLs from a JSON file containing movie data.

    Parameters:
    num_items (int): The maximum number of URLs to fetch.

    Returns:
    list: A list of URLs.
    """
    urls = []

    # Load movies from JSON file and add them to the list of URLs
    script_dir = os.path.dirname(__file__)  # Directory of the script
    file_path = os.path.join(script_dir, 'assets/movies.json')
    with open(file_path, encoding='utf-8') as user_file:
        file_contents = user_file.read()

    movies = json.loads(file_contents)
    max_items = num_items
    for i, movie in enumerate(movies):
        if i >= max_items:
            break
        urls.append("https://www.themoviedb.org/movie/" + str(movie.get('id')))
    return urls


def main():
    """
    Main function to load, process, and visualize documents.

    This function loads documents from URLs, transforms and cleans them,
    splits them into chunks, and adds them to a graph vector store.
    It also visualizes the documents as a text-based graph.
    """
    try:
        # Load and process documents
        loader = AsyncHtmlLoader(get_urls(num_items=1000))
        documents = loader.load()

        # Process documents in chunks of 10
        chunk_size = 10
        for i in range(0, len(documents), chunk_size):
            print(f"Processing documents {i + 1} to {i + chunk_size}...")
            document_chunk = documents[i:i + chunk_size]

            # Continue with the existing transformation and visualization
            transformer = LinkExtractorTransformer([
                #HtmlLinkExtractor().as_document_extractor(),
                KeybertLinkExtractor(),
            ])
            document_chunk = transformer.transform_documents(document_chunk)

            # Clean and preprocess documents using the new function
            document_chunk = clean_and_preprocess_documents(document_chunk)

            # The bs4 transformer is very capable and provides better content, but
            # it is much slower than the clean_and_preprocess_documents function used above.
            # For larger sets of documents, I tend to use the clean_and_preprocess_documents function.
            #bs4_transformer = BeautifulSoupTransformer()
            #document_chunk = bs4_transformer.transform_documents(document_chunk)

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=64,
            )
            document_chunk = text_splitter.split_documents(document_chunk)
            ner_extractor = GLiNERLinkExtractor(["Genre", "Topic"])
            transformer = LinkExtractorTransformer([ner_extractor])
            document_chunk = transformer.transform_documents(document_chunk)

            # Add documents to the graph vector store
            store.add_documents(document_chunk)

            # Visualize the graph text for the current chunk
            visualize_graph_text(document_chunk)

    except Exception as e:
        LOGGER.error("An error occurred: %s", e)


if __name__ == "__main__":
    main()
