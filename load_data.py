import os
import json
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
from util.config import logger, openai_api_key, astra_db_id, astra_token
from util.scrub import clean_and_preprocess_documents
from util.visualization import visualize_graph_text

# Initialize embeddings and LLM using OpenAI
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Initialize Astra connection using Cassio
cassio.init(database_id=astra_db_id, token=astra_token)
knowledge_store = CassandraGraphVectorStore(embeddings)


def get_urls(num_items=10):
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
    try:
        # Load and process documents
        loader = AsyncHtmlLoader(get_urls(num_items=10))
        raw_documents = loader.load()

        # Continue with the existing transformation and visualization
        transformer = LinkExtractorTransformer([
            HtmlLinkExtractor().as_document_extractor(),
            KeybertLinkExtractor(),
        ])
        transformed_documents = transformer.transform_documents(raw_documents)

        # Clean and preprocess documents using the new function
        cleaned_documents = clean_and_preprocess_documents(transformed_documents)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64
        )
        documents = text_splitter.split_documents(cleaned_documents)
        ner_extractor = GLiNERLinkExtractor(["Person", "Genre", "Location"])
        transformer = LinkExtractorTransformer([ner_extractor])
        documents = transformer.transform_documents(documents)

        # Add documents to the graph vector store
        knowledge_store.add_documents(documents)

        visualize_graph_text(documents)

    except Exception as e:
        logger.error("An error occurred: %s", e)

if __name__ == "__main__":
    main()
