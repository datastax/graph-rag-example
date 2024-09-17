def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

ANSWER_PROMPT = (
    "The original question is given below."
    "This question has been used to retrieve information from a vector store."
    "The matching results are shown below."
    "Use the information in the results to answer the original question.\n\n"
    "Original Question: {question}\n\n"
    "Vector Store Results:\n{context}\n\n"
    "Response:"
)