from colorama import Style, init
from main import ChainManager

# Initialize colorama
init(autoreset=True)

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