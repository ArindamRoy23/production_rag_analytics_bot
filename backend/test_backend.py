from app.utils.rag_functions import get_zero_shot_query
from app.utils.models import MessageData

# Test the get_zero_shot_query function
try:
    result = get_zero_shot_query(
        MessageData(
            query="What is the capital of France?",
            # chat_history=None,
            ollama_model_name="llama3.2:latest",
            temperature=0.5,
            top_p=0.9,
            max_tokens=100
        )
    )
    print("Success!")
    print("Question:", result.question)
    print("Answer:", result.answer)
    print("Retrieved contexts:", result.retrieved_contexts)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc() 