import os
import dspy
from dotenv import load_dotenv

load_dotenv()

# Get the Ollama base URL from the environment
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Remove trailing slash if present
if ollama_base_url.endswith('/'):
    ollama_base_url = ollama_base_url[:-1]

print(f"Using Ollama base URL: {ollama_base_url}")

# Create an Ollama LM
ollama_lm = dspy.OllamaLocal(
    model="llama3.2:latest",
    base_url=ollama_base_url,
    temperature=0.5,
    top_p=0.9,
    max_tokens=100,
)

# Try to generate a simple completion
try:
    with dspy.context(lm=ollama_lm):
        response = ollama_lm("What is the capital of France?")
    print("Response:", response)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc() 