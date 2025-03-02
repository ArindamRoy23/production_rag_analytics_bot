# Intelligent Dog Breed Assistant

## Introduction

This project is an intelligent question-answering system designed to help users find information about dog breeds. It leverages the [DSPy](https://github.com/stanfordnlp/dspy) framework for natural language processing, with a [FastAPI](https://github.com/tiangolo/fastapi) backend and [Streamlit](https://streamlit.io) frontend. The system runs entirely locally, using [Ollama](https://github.com/ollama/ollama) for language models, [Chroma DB](https://github.com/chroma-core/chroma) for vector storage, and [Arize Phoenix](https://github.com/Arize-ai/phoenix) for observability.

## Features

- **Comprehensive Dog Breed Information**: Access detailed information about various dog breeds including physical attributes, behavioral characteristics, care requirements, and more
- **Dual Query Processing**:
  - Natural Language Understanding: Handles semantic questions about breed recommendations and characteristics
  - Data Analytics: Processes numerical queries about breed statistics and comparisons
- **Interactive Web Interface**: Clean and functional Streamlit UI with conversation history
- **Advanced NLP Pipeline**: Powered by DSPy and Ollama
- **Efficient Data Storage**: Vector database implementation using Chroma DB
- **Performance Monitoring**: Real-time system monitoring with Arize Phoenix

## Architecture

The application combines several powerful technologies to create a robust dog breed information system:

- **DSPy Framework**: Powers the core NLP capabilities for understanding and processing breed-related queries
- **Ollama**: Provides the language model backend for natural language understanding
- **Chroma DB**: Enables efficient vector storage for quick breed information retrieval
- **Arize Phoenix**: Monitors system performance and query processing
- **FastAPI**: Handles API requests with a single endpoint for both NLP and analytical queries
- **Streamlit**: Delivers an intuitive user interface for interacting with the dog breed database

## API Specification

The system exposes an endpoint with the following interface:

```json
# Input Parameters:
{
    "user_id": string,  # Unique identifier for the user
    "query": string     # User's question about dog breeds
}

# Response:
{
    "answer": string    # Generated response to the query
}
```

## Installation

### Prerequisites

- Docker and Docker-Compose
- Git (optional, for cloning the repository)
- Ollama, follow the [readme](https://github.com/ollama/ollama) to set up and run a local Ollama instance.

### Dataset

The system uses a comprehensive dog breeds dataset that includes:
- Physical attributes (height, weight, lifespan)
- Behavioral characteristics and temperament
- Care requirements
- Training attributes
- Breed classifications
- Historical information
- Popularity metrics

The dataset can be accessed from [Kaggle Dog Breeds Dataset](https://www.kaggle.com/api/v1/datasets/download/mexwell/dog-breeds-dataset).

### Clone the Repository

First, clone the repository to your local machine (skip this step if you have the project files already).

```bash
git clone https://github.com/yourusername/dog-breed-assistant.git
cd dog-breed-assistant
```

### Getting Started with Local Development

#### Backend setup
First, navigate to the backend directory:
```bash
cd backend/
```

Second, setup the environment:

```bash
poetry config virtualenvs.in-project true
poetry install
poetry shell
```
Specify your environment variables in an .env file in backend directory.
Example .env file:
```yml
ENVIRONMENT=<your_environment_value>
INSTRUMENT_DSPY=<true or false>
COLLECTOR_ENDPOINT=<your_arize_phoenix_endpoint>
OLLAMA_BASE_URL=<your_ollama_instance_endpoint>
```
Third, run this command to create embeddings of data located in data/example folder:
```bash
python app/utils/load.py
```

Then run this command to start the FastAPI server:
```bash
python main.py
```

#### Frontend setup
First, navigate to the frontend directory:
```bash
cd frontend/
```

Second, setup the environment:

```bash
poetry config virtualenvs.in-project true
poetry install
poetry shell
```
Specify your environment variables in an .env file in backend directory.
Example .env file:
```yml
FASTAPI_BACKEND_URL = <your_fastapi_address>
```

Then run this command to start the Streamlit application:
```bash
streamlit run about.py
```

### Getting Started with Docker-Compose
This project now supports Docker Compose for easier setup and deployment, including backend services and Arize Phoenix for query tracing. 

1. Configure your environment variables in the .env file or modify the compose file directly.
2. Ensure that Docker is installed and running.
3. Run the command `docker-compose -f compose.yml up` to spin up services for the backend, and Phoenix.
4. Backend docs can be viewed using the [OpenAPI](http://0.0.0.0:8000/docs).
5. Frontend can be viewed using [Streamlit](http://0.0.0.0:8501)
6. Traces can be viewed using the [Phoenix UI](http://0.0.0.0:6006).
7. When you're finished, run `docker compose down` to spin down the services.

## Usage

The system can handle two types of queries:

### Natural Language Questions
Examples of supported natural language queries:
- "I have young kids and limited time for grooming. Which breed would suit my family?"
- "What breeds are known for being both protective and good with families?"
- "I'm looking for a tall, graceful dog with a flowing coat and independent personality"

### Analytical Queries
Examples of supported analytical queries:
- "List the 5 most popular breeds in the dataset"
- "Which breeds live the longest on average?"
- "Show me all large dogs (over 60cm) ordered by weight"


## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs, questions, and suggestions.

