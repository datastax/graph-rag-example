import cassio
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from langchain_community.document_transformers import Html2TextTransformer

from config import (
    logger,
    openai_api_key,
    astra_db_id,
    astra_token
)

from langchain_utils import (
    use_as_document_extractor,
    find_and_log_links,
    #use_link_extractor_transformer,
    #use_keybert_extractor,
    use_keybert_extract_one
)
from utils import format_docs, ANSWER_PROMPT

# Initialize embeddings and LLM using OpenAI
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(temperature=1, model_name="gpt-4o-mini")

# Initialize Astra connection using Cassio
cassio.init(database_id=astra_db_id, token=astra_token)

graph_vector_store = CassandraGraphVectorStore(embeddings)

class ChainManager:
    """
    Manages the setup and configuration of similarity and traversal chains
    for retrieving and processing documents using a graph vector store.
    """

    def __init__(self):
        """
        Initializes the ChainManager with placeholders for similarity and traversal chains.
        """
        self.similarity_chain = None
        self.traversal_chain = None

    def setup_chains(self):
        """
        Sets up the similarity and traversal chains using the graph vector store.
        
        The similarity chain retrieves documents based on vector similarity,
        while the traversal chain retrieves documents based on graph traversal.
        Both chains format the retrieved documents and use a language model to generate responses.
        """
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
                "score_threshold": 0.5,
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
            "https://www.themoviedb.org/movie/upcoming",
            "https://www.themoviedb.org/movie/698687-transformers-one",
            "https://www.themoviedb.org/movie/1087822-hellboy-the-crooked-man",
            "https://www.themoviedb.org/movie/1164355-lembayung",
            "https://www.themoviedb.org/movie/978796-bagman",
        ]

        # Load and process documents
        loader = AsyncHtmlLoader(urls)
        raw_documents = loader.load()
        use_as_document_extractor(raw_documents)
        #use_link_extractor_transformer(raw_documents)
        use_keybert_extract_one(raw_documents)
        #use_keybert_extractor(raw_documents)
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


async def get_similarity_result(chain_manager, question):
    """
    Gets the result from the similarity chain for a given question.
    
    Args:
        chain_manager (ChainManager): The chain manager instance.
        question (str): The question to be answered by the chain.
    """
    return chain_manager.similarity_chain.invoke(question).content


async def get_traversal_result(chain_manager, question):
    """
    Gets the result from the traversal chain for a given question.
    
    Args:
        chain_manager (ChainManager): The chain manager instance.
        question (str): The question to be answered by the chain.
    """
    return chain_manager.traversal_chain.invoke(question).content


if __name__ == "__main__":
    main()
