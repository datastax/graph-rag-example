# GraphRAG Example App
An example application that demonstrates how to use LangChain's graph vector stores to add structured data to RAG (Retrieval-Augmented Generation) applications. The app scrapes content from specified URLs, processes the content, and performs vector similarity and graph traversal searches.

```sh
  ____                 _     ____      _    ____ 
 / ___|_ __ __ _ _ __ | |__ |  _ \    / \  / ___|
| |  _| '__/ _` | '_ \| '_ \| |_) |  / _ \| |  _ 
| |_| | | | (_| | |_) | | | |  _ <  / ___ \ |_| |
 \____|_|  \__,_| .__/|_| |_|_| \_\/_/   \_\____|
                |_|                                           
                        *no graph database needed!!!
```

## Installation

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
	- Copy the [`.env.example`](./.env.example") file to `.env`:
	  ```sh
	  cp .env.example .env
	  ```
	- Fill in the required environment variables in the `.env` file.

## Launch the App

1. **Run the main script**:
	```sh
	python app.py
	```

This will start the application, scrape the specified URLs, process the content, and perform vector similarity and graph traversal searches.

## License

This project is licensed under the MIT License.

