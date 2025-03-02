import os
import dspy
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

# Get the Ollama base URL from the environment
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Remove trailing slash if present
if ollama_base_url.endswith('/'):
    ollama_base_url = ollama_base_url[:-1]

print(f"Using Ollama base URL: {ollama_base_url}")

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

# Try to use the RAG module
try:
    # Create an Ollama LM
    ollama_lm = dspy.OllamaLocal(
        model="llama3.2:latest",
        base_url=ollama_base_url,
        temperature=0.5,
        top_p=0.9,
        max_tokens=100,
    )
    
    # Create a Passage class with the required attributes
    @dataclass
    class Passage:
        long_text: str
        
    # Create a dummy retriever model that returns passages in the correct format
    class DummyRetriever:
        def __call__(self, query, k=5, **kwargs):
            passages = [Passage(long_text="Paris is the capital of France.")]
            return passages
    
    # Configure DSPy to use the dummy retriever
    dspy.settings.configure(rm=DummyRetriever())
    
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