"""
This module sets up and manages chains for retrieving 
and processing documents using a graph vector store.
It supports three types of searches: 
    vector similarity, graph traversal, and Maximal Marginal Relevance (MMR).

The main components of the module are:
- Initialization of embeddings and language model using OpenAI.
- Initialization of the DataStax Astra DB connection using Cassio.
- Definition of the ChainManager class to manage the setup and configuration of the chains.
- Functions to get results from the similarity, traversal, and MMR chains.

"""
import cassio
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from util.config import OPENAI_API_KEY, ASTRA_DB_ID, ASTRA_TOKEN, MOVIE_NODE_TABLE, ANSWER_PROMPT

# Initialize embeddings and LLM using OpenAI
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=1, model_name="gpt-4o-mini")

# Initialize Astra connection using Cassio
cassio.init(database_id=ASTRA_DB_ID, token=ASTRA_TOKEN)
store = CassandraGraphVectorStore(embeddings, node_table=MOVIE_NODE_TABLE)

class ChainManager:
    """
    Manages the setup and configuration of similarity and traversal chains
    for retrieving and processing documents using a graph vector store.
    """
    def __init__(self):
        self.similarity_chain = None
        self.mmr_chain = None
        self.mmr_retriever = None
        self.similarity_retriever = None

    def format_docs(self, docs):
        """
        Formats documents by concatenating their content.

        Parameters:
        docs (list): List of documents to format.

        Returns:
        str: Concatenated content of the documents.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def setup_chains(self, k=10, depth=3, lambda_mult=0.25):
        """
        Sets up the similarity and traversal chains using the graph vector store.
        
        The similarity chain retrieves documents based on vector similarity,
        while the traversal chain retrieves documents based on graph traversal.
        Both chains format the retrieved documents and use a language model to generate responses.

        Parameters:
        k (int): The number of top results to retrieve.
        depth (int): The depth of the graph traversal.
        lambda_mult (float): The lambda multiplier for MMR.
        """
        self.similarity_retriever = store.as_retriever(
            search_type="similarity", search_kwargs={
                "k": k, # Return the top results
                "depth": 0,
            })

        self.mmr_retriever = store.as_retriever(
            search_type="mmr_traversal", search_kwargs={
                "k": k, # top k results
                "depth": depth,
                "lambda_mult": lambda_mult, # 0 = more diverse, 1 = more relevant
                #"fetch_k": 50
            })

        self.similarity_chain = (
            {"context": self.similarity_retriever | self.format_docs, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_messages([ANSWER_PROMPT])
            | llm
        )
        self.mmr_chain = (
            {"context": self.mmr_retriever | self.format_docs, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_messages([ANSWER_PROMPT])
            | llm
        )

async def get_similarity_result(chain_manager, question):
    """
    Gets the result from the similarity chain for a given question.
    
    Args:
        chain_manager (ChainManager): The chain manager instance.
        question (str): The question to be answered by the chain.

    Returns:
        tuple: A tuple containing the similarity result and usage metadata.
    """
    invoked_chain = chain_manager.similarity_chain.invoke(question)
    content = invoked_chain.content
    usage_metadata = invoked_chain.usage_metadata
    return content, usage_metadata


async def get_mmr_result(chain_manager, question):
    """
    Gets the result from the mmr chain for a given question.
    
    Args:
        chain_manager (ChainManager): The chain manager instance.
        question (str): The question to be answered by the chain.
    
    Returns:
        tuple: A tuple containing the MMR result and usage metadata.
    """
    invoked_chain = chain_manager.mmr_chain.invoke(question)
    content = invoked_chain.content
    usage_metadata = invoked_chain.usage_metadata
    return content, usage_metadata
