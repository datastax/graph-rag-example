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
from util.config import openai_api_key, astra_db_id, astra_token, ANSWER_PROMPT

# Initialize embeddings and LLM using OpenAI
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(temperature=1, model_name="gpt-4o-mini")

# Initialize Astra connection using Cassio
cassio.init(database_id=astra_db_id, token=astra_token)
knowledge_store = CassandraGraphVectorStore(embeddings)

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
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def setup_chains(self):
        """
        Sets up the similarity and traversal chains using the graph vector store.
        
        The similarity chain retrieves documents based on vector similarity,
        while the traversal chain retrieves documents based on graph traversal.
        Both chains format the retrieved documents and use a language model to generate responses.
        """
        self.similarity_retriever = knowledge_store.as_retriever(
            search_type="similarity", search_kwargs={
                "k": 10, # Return the top results
                "depth": 0,
            })

        self.mmr_retriever = knowledge_store.as_retriever(
            search_type="mmr_traversal", search_kwargs={
                "k": 10, # top k results
                "depth": 3,
                "lambda_mult": 0.25, # 0 = more diverse, 1 = more relevant
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
        tuple: A tuple containing the traversal result and the traversal path.
    """
    invoked_chain = chain_manager.mmr_chain.invoke(question)
    content = invoked_chain.content
    usage_metadata = invoked_chain.usage_metadata
    return content, usage_metadata
