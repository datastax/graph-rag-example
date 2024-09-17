import logging
import os
import coloredlogs
import cassio
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from langchain_community.document_transformers import Html2TextTransformer

# Import utility functions
from utility import (
    find_and_log_links,
    use_as_document_extractor,
    #use_link_extractor_transformer,
    #log_pretty_list,
    use_keybert_extractor,
)

# Load environment variables from .env file
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

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

# Log the ASCII art
logger.info(ASCII_ART)

# Initialize Astra connection using Cassio
astra_db_id = os.getenv("ASTRA_DB_DATABASE_ID")
astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
astra_endpoint = os.getenv("ASTRA_DB_ENDPOINT")

if not all([astra_db_id, astra_token, astra_endpoint]):
    raise ValueError("Astra DB credentials must be set.")

cassio.init(database_id=astra_db_id, token=astra_token)

# Initialize embeddings and graph vector store
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")
embeddings = OpenAIEmbeddings(api_key=api_key)
graph_vector_store = CassandraGraphVectorStore(embeddings)

# Main function to process the URL
def main():
    try:
        urls = [
                "https://python.langchain.com/v0.2/docs/integrations/providers/astradb/",
                "https://docs.datastax.com/en/astra/home/astra.html",
                ]

        # Load documents from URLs and extract links
        loader = AsyncHtmlLoader(urls)
        raw_documents = loader.load()

        use_as_document_extractor(raw_documents)
        use_keybert_extractor(raw_documents)
        find_and_log_links(raw_documents)

        # Transform documents to text
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(raw_documents)

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(docs_transformed)

        # Add documents to the graph vector store
        graph_vector_store.add_documents(documents)

        # Vector Searches
        query="How do Astra and Langchain work together?"

        # Perform a vector similarity search
        #vector_search_result = graph_vector_store.similarity_search(
        #    query=query,
        #    k=1, depth=1
        #)
        #for result in vector_search_result:
        #    print("Similiarity Search Result Titles: ", result)
        #log_pretty_list(vector_search_result, "Vector Search Result Titles")

        # Perform a graph traversal search
        traversal_search_result = graph_vector_store.traversal_search(
            query=query,
            k=1
        )
        for result in traversal_search_result:
            print("\n\nTraversal Search Result Titles: ", result)
        #log_pretty_list(traversal_search_result, "Traversal Search Result Titles")

    except Exception as e:
        print(f"An error occurred: {e}")

# Entry point for the script
if __name__ == "__main__":
    main()
