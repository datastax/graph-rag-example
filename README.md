# 🚀 GraphRAG Example App
An example application that demonstrates how to use LangChain's [graph_vectorstores](https://python.langchain.com/v0.2/api_reference/community/graph_vectorstores.html#) and [CassandraGraphVectorStore](https://python.langchain.com/v0.2/api_reference/community/graph_vectorstores/langchain_community.graph_vectorstores.cassandra.CassandraGraphVectorStore.html) to add structured data to RAG (Retrieval-Augmented Generation) applications. The app scrapes content from specified URLs, processes the content, and performs vector similarity and graph traversal searches.

```sh
  ____                 _     ____      _    ____ 
 / ___|_ __ __ _ _ __ | |__ |  _ \    / \  / ___|
| |  _| '__/ _` | '_ \| '_ \| |_) |  / _ \| |  _ 
| |_| | | | (_| | |_) | | | |  _ <  / ___ \ |_| |
 \____|_|  \__,_| .__/|_| |_|_| \_\/_/   \_\____|
                |_|                                           
                        *no graph database needed!!!
```

## 📦 Installation

1. **Clone the repository**:
	```sh
	git clone https://github.com/datastaxdevs/graph-rag-example.git
	cd graphRAG_example
	```

2. **Create and activate a virtual environment**:
	```sh
	python3 -m venv venv
	source venv/bin/activate
	```

3. **Install the required dependencies**:
	```sh
	pip install -r requirements.txt
	```

4. **Set up the environment variables**:
	- Copy the [`.env.example`](.env.example) file to `.env`:
	  ```sh
	  cp .env.example .env
	  ```
	- Fill in the required environment variables in the `.env` file.

	Once you have your .env ready, create a [DataStax Astra Vector database](https://docs.datastax.com/en/astra-db-serverless/get-started/quickstart.html) if you don't already have one and copy the database ID, API endpoint, and an application token from the database overview page. Everything you need will be there.

	You also need an [OpenAI API key](https://platform.openai.com/api-keys) to power the LLM responsible for giving responses.

## 🚀 Launch the App


1. **Run the data loading script**:
	```sh
	python load_data.py
	```
	*load_data.py pulls data from [www.themoviedb.org](www.themoviedb.org) and extracts page content and metadata used in the graph.*

2. **Run the main script**:
	```sh
	python app.py
	```
	*app.py displays a [Dash](https://dash.plotly.com/) based UI that allows a real-time comparison between both similarity and traversal based searches using graph RAG.*


## 📜 License

This project is licensed under the MIT License.

