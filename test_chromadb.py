import os
import dspy
from dotenv import load_dotenv
from dspy.retrieve.chromadb_rm import ChromadbRM
from chromadb import EmbeddingFunction, Documents, Embeddings
import ollama

load_dotenv()

# Get the Ollama base URL from the environment
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Remove trailing slash if present
if ollama_base_url.endswith('/'):
    ollama_base_url = ollama_base_url[:-1]

print(f"Using Ollama base URL: {ollama_base_url}")

# Custom Embedding function that supports Ollama embeddings
class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    This class is used to get embeddings for a list of texts using Ollama Python Library.
    It requires a host url and a model name. The default model name is "nomic-embed-text".
    """

    def __init__(
        self, host: str = "http://localhost:11434", model_name: str = "nomic-embed-text"
    ):
        self._client = ollama.Client(host)
        self._model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        """
        Get the embeddings for a list of texts.
        """
        embeddings = []
        # Call Ollama Embedding API for each document.
        for document in input:
            embedding = self._client.embeddings(model=self._model_name, prompt=document)
            embeddings.append(embedding["embedding"])

        return embeddings

# Define a simple signature
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Define a simple RAG module
class RAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# Try to use the RAG module with ChromadbRM
try:
    # Create the embedding function
    ollama_embedding_function = OllamaEmbeddingFunction(host=ollama_base_url)
    
    # Create the retriever model
    DATA_DIR = "data"
    retriever_model = ChromadbRM(
        "quickstart",
        f"{DATA_DIR}/chroma_db",
        embedding_function=ollama_embedding_function,
        k=5,
    )
    
    # Configure DSPy to use the retriever model
    dspy.settings.configure(rm=retriever_model)
    
    # Create an Ollama LM
    ollama_lm = dspy.OllamaLocal(
        model="llama3.2:latest",
        base_url=ollama_base_url,
        temperature=0.5,
        top_p=0.9,
        max_tokens=100,
    )
    
    # Create a RAG instance
    rag = RAG()
    
    # Use the RAG instance
    with dspy.context(lm=ollama_lm):
        pred = rag(question="What is the capital of France?")
    
    print("Question:", "What is the capital of France?")
    print("Answer:", pred.answer)
    print("Context:", pred.context)
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc() 