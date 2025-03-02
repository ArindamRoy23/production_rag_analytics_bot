"""Pydantic models."""

from pydantic import BaseModel
from typing import List, Optional
import dspy

class MessageData(BaseModel):
    """Datamodel for messages."""

    query: str
    # chat_history: List[dict] | None
    ollama_model_name: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1000
    pipeline_id: Optional[int] = None


class RAGResponse(BaseModel):
    """Datamodel for RAG response."""

    question: str
    answer: str
    retrieved_contexts: List[str]


class QAItem(BaseModel):
    question: str
    answer: str


class QAList(BaseModel):
    """Datamodel for trainset."""

    items: List[QAItem]
    ollama_model_name: str
    temperature: float
    top_p: float
    max_tokens: int

# # Define a DSPy signature for analytical answers
# class GenerateAnalyticalAnswer(dspy.Signature):
#     """Generate answers for analytical questions about dog breeds."""
    
#     data_summary = dspy.InputField(desc="summarized data from analysis")
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="Comprehensive answer based on data analysis")

# Create a module for analytics
class DataAnalytics(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateAnalyticalAnswer)
    
    def forward(self, question: str, data_summary: str):
        prediction = self.generate_answer(
            data_summary=data_summary,
            question=question
        )
        return dspy.Prediction(answer=prediction.answer)   
    


# Create a DSPy module for generating the data analysis prompt with system instructions
class DataAnalysisPrompt(dspy.Signature):
    question = dspy.InputField(desc="User's analytical question")
    columns_info = dspy.InputField(desc="Available dataframe columns")
    pandas_code = dspy.OutputField(desc="Python pandas code snippet to answer the question")


# Define a DSPy signature for analytical answers with system instructions
class GenerateAnalyticalAnswer(dspy.Signature):
    """Generate answers for analytical questions about dog breeds."""
    
    # system_prompt = dspy.InputField(desc="Instructions for how to generate the answer")
    data_summary = dspy.InputField(desc="summarized data from analysis")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Comprehensive answer based on data analysis")