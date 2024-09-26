from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean


def clean_and_preprocess_documents(documents):
    """
    Cleans and preprocesses a list of documents using unstructured.

    Parameters:
    documents (list): List of documents to clean and preprocess.

    Returns:
    list: List of cleaned and preprocessed documents.
    """
    cleaned_documents = []
    for doc in documents:
        # Partition the HTML content
        elements = partition_html(text=doc.page_content)
        # Clean the text content
        cleaned_text = clean(" ".join([element.text for element in elements]))
        # Update the document content with cleaned text
        doc.page_content = scrub(cleaned_text)
        cleaned_documents.append(doc)
    return cleaned_documents


def scrub(content):
  content = content.replace("What's your", "")
  content = content.replace("Login to use TMDB's new rating system.", "")
  content = content.replace("Welcome to Vibes, TMDB's new rating system! For more information, visit the  contribution bible.", "")
  content = content.replace("Looks like we're missing the following data in en-US or en-US...", "")
  content = content.replace("Login to edit", "")
  content = content.replace("Login to report an issue", "")
  return content