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
from langchain_community.document_transformers import Html2TextTransformer
from util.config import logger, openai_api_key, astra_db_id, astra_token
from util.visualization import find_and_log_links

# Initialize embeddings and LLM using OpenAI
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Initialize Astra connection using Cassio
cassio.init(database_id=astra_db_id, token=astra_token)
knowledge_store = CassandraGraphVectorStore(embeddings)

def main():
    try:
        urls = [
            "https://www.themoviedb.org/movie/upcoming",
            "https://www.themoviedb.org/movie/698687-transformers-one",
            "https://www.themoviedb.org/movie/1087822-hellboy-the-crooked-man",
            "https://www.themoviedb.org/movie/1164355-lembayung",
            "https://www.themoviedb.org/movie/978796-bagman",
        ]

        # Load movies from JSON file and add them to the list of URLs
        script_dir = os.path.dirname(__file__)  # Directory of the script
        file_path = os.path.join(script_dir, 'assets/movies.json')
        with open(file_path, encoding='utf-8') as user_file:
            file_contents = user_file.read()

        movies = json.loads(file_contents)
        max_items = 10
        for i, movie in enumerate(movies):
            if i >= max_items:
                break
            urls.append("https://www.themoviedb.org/movie/" + str(movie.get('id')))

        # Load and process documents
        loader = AsyncHtmlLoader(urls)
        raw_documents = loader.load()

        transformer = LinkExtractorTransformer([
            HtmlLinkExtractor().as_document_extractor(),
            KeybertLinkExtractor(),
            #GLiNERLinkExtractor(["Title", "Genre", "Date", "Trailer"]),
        ])
        raw_documents = transformer.transform_documents(raw_documents)
        find_and_log_links(raw_documents)

        # Transform documents to text
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(raw_documents)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64
        )
        documents = text_splitter.split_documents(docs_transformed)

        # Add documents to the graph vector store
        knowledge_store.add_documents(documents)

    except Exception as e:
        logger.error("An error occurred: %s", e)

if __name__ == "__main__":
    main()
