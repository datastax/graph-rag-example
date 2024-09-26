"""
This module configures the environment and logging for the application, and initializes
necessary credentials and constants for OpenAI and Astra DB connections.

- Loads environment variables from a .env file using `dotenv`.
- Configures a logger with colored logs for better readability.
- Retrieves and validates the OpenAI API key from environment variables.
- Retrieves and validates Astra DB credentials (database ID, application token, and endpoint) 
    from environment variables.
- Defines a constant `ANSWER_PROMPT` used for formatting responses based on vector store results.

Raises:
    ValueError: If the required environment variables for 
    OpenAI API key or Astra DB credentials are not set.
"""
import logging
import os
import coloredlogs
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set debug mode, `True` will generate graphs in multiple
# formats (dot, png, text) for use in analyzing results
DEBUG_MODE=True

# Configure logger
LOGGER = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=LOGGER)

# Initialize embeddings and LLM using OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

# Initialize Astra connection using Cassio
ASTRA_DB_ID = os.getenv("ASTRA_DB_DATABASE_ID")
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_ENDPOINT")
if not all([ASTRA_DB_ID, ASTRA_TOKEN, ASTRA_ENDPOINT]):
    raise ValueError("Astra DB credentials must be set.")

ANSWER_PROMPT = (
    "The original question is given below."
    "This question has been used to retrieve information from a vector store."
    "The matching results are shown below."
    "Only use context to answer the question."
    "Use the information in the results to answer the original question."
    "Give responses in pretty Markdown."
    "Original Question: {question}\n\n"
    "Vector Store Results:\n{context}\n\n"
    "Response:"
)
