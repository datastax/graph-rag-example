import cassio
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from langchain_community.document_transformers import Html2TextTransformer
from colorama import Style, init

from config import (
    logger,
    openai_api_key,
    astra_db_id,
    astra_token,
    ASCII_ART
)
from langchain_utils import find_and_log_links, use_as_document_extractor
from utils import format_docs, ANSWER_PROMPT

# Initialize colorama
init(autoreset=True)

print(ASCII_ART)

# Initialize embeddings and LLM using OpenAI
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(temperature=1, model_name="gpt-4o-mini")

# Initialize Astra connection using Cassio
cassio.init(database_id=astra_db_id, token=astra_token)

graph_vector_store = CassandraGraphVectorStore(embeddings)

class ChainManager:
    def __init__(self):
        self.similarity_chain = None
        self.traversal_chain = None

    def setup_chains(self):
        # Set up retrievers
        similarity_retriever = graph_vector_store.as_retriever(
            search_kwargs={
                "k": 1, 
                "depth": 0
            })
        traversal_retriever = graph_vector_store.as_retriever(
            search_type="traversal", search_kwargs={
                "k": 10, 
                "depth": 1,
                "score_threshold": 0.2,
            })

        # Set up chains
        self.similarity_chain = (
            {"context": similarity_retriever | format_docs, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_messages([ANSWER_PROMPT])
            | llm
        )
        self.traversal_chain = (
            {"context": traversal_retriever | format_docs, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_messages([ANSWER_PROMPT])
            | llm
        )

def main():
    try:
        urls = [
            "https://python.langchain.com/v0.2/docs/integrations/providers/astradb/",
            "https://docs.datastax.com/en/astra/home/astra.html",
            "https://github.com/langflow-ai/langflow",
            "https://www.langchain.com/",
            "https://docs.langflow.org/integrations-langsmith",
            "https://python.langchain.com/v0.2/api_reference/community/graph_vectorstores.html",
            "https://python.langchain.com/v0.2/api_reference/community/graph_vectorstores/langchain_community.graph_vectorstores.cassandra.CassandraGraphVectorStore.html",
        ]

        # Load and process documents
        loader = AsyncHtmlLoader(urls)
        raw_documents = loader.load()
        use_as_document_extractor(raw_documents)
        find_and_log_links(raw_documents)

        # Transform documents to text
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(raw_documents)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        documents = text_splitter.split_documents(docs_transformed)

        # Add documents to the graph vector store
        graph_vector_store.add_documents(documents)

    except Exception as e:
        logger.error("An error occurred: %s", e)


def compare_results(question):
    print(Style.BRIGHT + "\nQuestion:")
    print(Style.NORMAL + question)

    # Initialize ChainManager and set up chains
    chain_manager = ChainManager()
    chain_manager.setup_chains()
    
    output_answer = chain_manager.similarity_chain.invoke(question)
    print(Style.BRIGHT + "\n\nVector Similarity Result:")
    print(Style.NORMAL + output_answer.content)

    output_answer = chain_manager.traversal_chain.invoke(question)
    print(Style.BRIGHT + "\n\nGRAPH Traversal Result:")
    print(Style.NORMAL + output_answer.content)


if __name__ == "__main__":
    main()
    compare_results("How do I setup a graph vector store using Astra?")
