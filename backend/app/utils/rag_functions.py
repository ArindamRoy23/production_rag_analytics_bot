"""DSPy functions."""

import os
import pandas as pd

import dspy
import ollama
from dotenv import load_dotenv
from dspy.retrieve.chromadb_rm import ChromadbRM
from dspy.teleprompt import BootstrapFewShot

from app.utils.load import OllamaEmbeddingFunction
from app.utils.rag_modules import RAG
from app.utils.models import MessageData, RAGResponse, QAList

from app.utils.models import DataAnalytics, DataAnalysisPrompt, GenerateAnalyticalAnswer
from app.utils.helper import extract_python_code

load_dotenv()


from typing import Dict

# Global settings
DATA_DIR = "data"
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "localhost")
ollama_embedding_function = OllamaEmbeddingFunction(host=ollama_base_url)

retriever_model = ChromadbRM(
    "dogs",
    f"{DATA_DIR}/chroma_db",
    embedding_function=ollama_embedding_function,
    k=5,
)

dspy.settings.configure(rm=retriever_model)


def get_zero_shot_query(payload: MessageData):
    rag = RAG()
    # Global settings
    ollama_lm = dspy.OllamaLocal(
        model=payload.ollama_model_name,
        base_url=ollama_base_url,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )
    # parsed_chat_history = ", ".join(
    #     [f"{chat['role']}: {chat['content']}" for chat in payload.chat_history]
    # )
    with dspy.context(lm=ollama_lm):
        pred = rag(
            # question=parsed_chat_history +  "user: " + payload.query, 
            question=payload.query, 

        )

    return RAGResponse(
        question=payload.query,
        answer=pred.answer,
        retrieved_contexts=[c[:200] + "..." for c in pred.context],
    )


def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM


def compile_rag(qa_items: QAList) -> Dict:
    # Global settings
    ollama_lm = dspy.OllamaLocal(
        model=qa_items.ollama_model_name,
        base_url=ollama_base_url,
        temperature=qa_items.temperature,
        top_p=qa_items.top_p,
        max_tokens=qa_items.max_tokens,
    )

    trainset = [
        dspy.Example(
            question=item.question,
            answer=item.answer,
        ).with_inputs("question")
        for item in qa_items.items
    ]

    # Set up a basic teleprompter, which will compile our RAG program.
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

    # Compile!
    with dspy.context(lm=ollama_lm):
        compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

    # Saving
    compiled_rag.save(f"{DATA_DIR}/compiled_rag.json")

    return {"message": "Successfully compiled RAG program!"}


def get_compiled_rag(payload: MessageData):
    # Loading:
    rag = RAG()
    rag.load(f"{DATA_DIR}/compiled_rag.json")

    # Global settings
    ollama_lm = dspy.OllamaLocal(
        model=payload.ollama_model_name,
        base_url=ollama_base_url,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )
    # parsed_chat_history = ", ".join(
    #     [f"{chat['role']}: {chat['content']}" for chat in payload.chat_history]
    # )
    with dspy.context(lm=ollama_lm):
        pred = rag(
            question=payload.query,  # chat_history=parsed_chat_history
        )

    return RAGResponse(
        question=payload.query,
        answer=pred.answer,
        retrieved_contexts=[c[:200] + "..." for c in pred.context],
    )


def get_list_ollama_models():
    # Use OLLAMA_HOST if available, otherwise fall back to OLLAMA_BASE_URL
    host = os.getenv("OLLAMA_HOST", ollama_base_url)
    # Remove trailing slash if present
    if host.endswith('/'):
        host = host[:-1]
    client = ollama.Client(host=host)

    models = []
    try:
        models_list = client.list()
        for model in models_list["models"]:
            models.append(model["name"])
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return {"models": [], "error": str(e)}

    return {"models": models}


def get_pipeline_id(payload: MessageData):
    import chromadb
    
    # Get ollama embedding for the query
    ollama_ef = OllamaEmbeddingFunction(host=ollama_base_url)
    query_embedding = ollama_ef(payload.query)
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=f"{DATA_DIR}/chroma_db")
    
    # Get the collection
    collection = chroma_client.get_collection("pipeline_description")
    
    # Query the collection to get nearest neighbor
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1,
        include=['distances', 'documents', 'metadatas']
    )
    
    # Extract the ID and distance of the closest match
    if results and len(results['ids']) > 0:
        closest_id = results['ids'][0][0]  # First item's ID
        distance = results['distances'][0][0]  # First item's distance
        return int(closest_id)
    
    return closest_id

def execute_pipeline(payload: MessageData, pipeline_id: int|None=None):
    if pipeline_id is None:
        pipeline_id = get_pipeline_id(payload)
    if pipeline_id == 1:
        return pipeline_nlp(payload)
    elif pipeline_id == 2:
        return pipeline_analytics(payload)
    else:
        return "Invalid pipeline ID"

def pipeline_nlp(payload: MessageData):
    return get_zero_shot_query(payload)

def pipeline_analytics(payload: MessageData):
    """
    Process analytical queries about dog breed data using system prompts and few-shot prompting.
    
    This function:
    1. Loads the dog breed dataframe
    2. Creates a summary of the dataframe based on the query using system prompts
    3. Uses few-shot examples to guide the analysis
    4. Sends this summary with the query to an LLM
    5. Returns a formatted response
    """

    # Load the dog breed dataframe
    try:
        df = pd.read_csv("data/akc-data-latest.csv").rename(columns={'Unnamed: 0': 'breed_name'})
    except Exception as e:
        return RAGResponse(
            question=payload.query,
            answer=f"Error loading dog breed data: {str(e)}",
            retrieved_contexts=[]
        )

    # Initialize Ollama LM
    ollama_lm = dspy.OllamaLocal(
        model=payload.ollama_model_name,
        base_url=ollama_base_url,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )

    # Get column information for context
    columns_info = "\n".join([f"- {col} ({df[col].dtype})" for col in df.columns])

    # Define system prompt clearly
    analysis_system_prompt = (
        "You are a data analyst specialized in dog breed analytics. Your task is to generate Python code using pandas "
        "to analyze a dog breed dataframe based on the user's question. Your code should:\n"
        "1. Be concise and efficient\n"
        "2. Include proper variable naming\n"
        "3. Store the final result in a variable called 'result'\n"
        "4. Add comments explaining key steps\n"
        "5. Focus on the specific analytics requested in the question"
    )

    # Define few-shot examples clearly
    few_shot_examples = (
        "Example 1:\n"
        "Question: What are the top 5 heaviest dog breeds?\n"
        "Code:\n"
        "```python\n"
        "result = df.sort_values(by='weight', ascending=False).head(5)\n"
        "result = result[['breed_name', 'weight']]\n"
        "```\n\n"
        "Example 2:\n"
        "Question: What's the average lifespan of small dog breeds?\n"
        "Code:\n"
        "```python\n"
        "small_dogs = df[df['weight'] < 30]\n"
        "avg_lifespan = small_dogs['lifespan'].mean()\n"
        "result = f'The average lifespan of small dogs is {avg_lifespan:.2f} years'\n"
        "```\n\n"
        "Example 3:\n"
        "Question: Compare the trainability of working dogs versus toy breeds\n"
        "Code:\n"
        "```python\n"
        "working_dogs = df[df['group'] == 'Working']\n"
        "toy_dogs = df[df['group'] == 'Toy']\n"
        "working_avg = working_dogs['trainability'].mean()\n"
        "toy_avg = toy_dogs['trainability'].mean()\n"
        "result = pd.DataFrame({\n"
        "    'Group': ['Working', 'Toy'],\n"
        "    'Average Trainability': [working_avg, toy_avg]\n"
        "})\n"
        "```"
    )

    # Correctly instantiate and call DSPy Predict
    with dspy.context(lm=ollama_lm):
        analysis_result = dspy.Predict(DataAnalysisPrompt)(
            question=f"Write pandas code to answer: {payload.query}",
            columns_info=f'''You have an initiated datyaframe named "df". 
                        The dataframe has these columns: {columns_info}. "
                         "Write concise pandas code. Store the final result in a variable named 'result'. "
                         "Do not include explanations or markdown formatting, only provide executable Python code.'''
        )

    # Extract and execute the generated pandas code safely
    try:
        local_vars = {"df": df, "pd": pd}
        code = extract_python_code(analysis_result.pandas_code)

        # Execute the pandas analysis code
        exec(code, globals(), local_vars)

        # Get the result from the executed code
        data_summary = str(local_vars.get('result', "No result variable found."))
    except Exception as e:
        data_summary = f"Error in data analysis: {str(e)}"

    # Generate the final answer using DataAnalytics module
    answer_system_prompt = (
        "You are a dog breed expert and data analyst. Your task is to provide a comprehensive, well-formatted answer "
        "to the user's analytical question about dog breeds. Your response should:\n"
        "1. Be conversational and engaging\n"
        "2. Include specific numerical insights from the data\n"
        "3. Explain the methodology of analysis when relevant\n"
        "4. Provide context about the dog breeds mentioned\n"
        "5. Highlight any limitations in the data or analysis"
    )
    analytics = DataAnalytics()
    with dspy.context(lm=ollama_lm):
        pred = analytics(
            question=payload.query,
            data_summary=data_summary
        )

    return RAGResponse(
        question=payload.query,
        answer=pred.answer,
        retrieved_contexts=[pred.answer]
    )

if __name__ == "__main__":
    r = execute_pipeline(MessageData(query="List the 5 most popular breeds in the dataset. Reply like a helpful mckinsey data analyst. \
                                     Do not respond if the data errored out. ",
                                    # chat_history=[{"role": "user", "content": "I want to buy a dog. What is the best breed for me?"},
                                    #               {"role": "assistant", "content": "Tell me about your lifestyle and preferences. \
                                    #                I will recommend the best breed for you."}],
                                    ollama_model_name="llama3.2:latest", 
                                    temperature=0.5, 
                                    top_p=0.9, 
                                    max_tokens=1000),
                                    pipeline_id=2)
