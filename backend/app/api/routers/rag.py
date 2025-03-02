"""Endpoints."""

from fastapi import APIRouter
from app.utils.models import MessageData, RAGResponse, QAList
from app.utils.rag_functions import (
    get_zero_shot_query,
    get_compiled_rag,
    compile_rag,
    get_list_ollama_models,
    execute_pipeline
)

rag_router = APIRouter()


# @rag_router.get("/healthcheck")
# async def healthcheck():

#     return {"message": "Thanks for playing."}


@rag_router.get("/list-models")
async def list_models():
    return get_list_ollama_models()


@rag_router.post("/zero-shot-query", response_model=RAGResponse)
async def zero_shot_query(payload: MessageData):
    return get_zero_shot_query(payload=payload)


# @rag_router.post("/compiled-query", response_model=RAGResponse)
# async def compiled_query(payload: MessageData):
#     return get_compiled_rag(payload=payload)


# @rag_router.post("/compile-program")
# async def compile_program(qa_list: QAList):

#     print(qa_list)
#     return compile_rag(qa_items=qa_list)


@rag_router.post("/execute-pipeline", response_model=RAGResponse)
async def pipeline_query(payload: MessageData):
    """
    Execute the appropriate pipeline based on the pipeline_id.
    If no pipeline_id is provided, it will be automatically determined.
    """
    pipeline_id = getattr(payload, 'pipeline_id', None)
    return execute_pipeline(payload=payload, pipeline_id=pipeline_id)


@rag_router.post("/api/query", response_model=RAGResponse)
async def pipeline_query(user_id: str, query: str):
    """
    Execute the appropriate pipeline based on the query type.
    
    Args:
        user_id (str): Unique identifier for the user
        query (str): User's question about dog breeds
    
    Returns:
        RAGResponse: Contains the answer and any retrieved contexts
    """
    # Create MessageData with default model settings
    payload = MessageData(query=query,
                                    # chat_history=[{"role": "user", "content": "I want to buy a dog. What is the best breed for me?"},
                                    #               {"role": "assistant", "content": "Tell me about your lifestyle and preferences. \
                                    #                I will recommend the best breed for you."}],
                                    ollama_model_name="llama3.2:latest", 
                                    temperature=0.5, 
                                    top_p=0.9, 
                                    max_tokens=1000)
    
    pipeline_id = getattr(payload, 'pipeline_id', 1)
    return execute_pipeline(payload=payload, pipeline_id=pipeline_id)
