import streamlit as st

st.set_page_config(
    page_title="Intelligent Dog Breed Assistant",
    page_icon="🐕",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Intelligent Dog Breed Assistant")

st.sidebar.info("Select a demo above.")

st.markdown(
    """
    ## Description
    This [Streamlit](https://streamlit.io) application serves as an intelligent question-answering system designed to help users find information about dog breeds. 
    Powered by the [DSPy](https://github.com/stanfordnlp/dspy) framework and backed by [FastAPI](https://github.com/tiangolo/fastapi), 
    it provides comprehensive information about various dog breeds through both natural language understanding and data analytics capabilities.

    ### Key Features
    - **Natural Language Understanding**: Ask questions about breed recommendations and characteristics
    - **Data Analytics**: Get insights about breed statistics and comparisons
    - **Comprehensive Dataset**: Access detailed information about:
        - Physical attributes
        - Behavioral characteristics
        - Care requirements
        - Training attributes
        - Breed classifications
        - Historical information
        - Popularity metrics

    ### Technology Stack
    - **Backend**: FastAPI with DSPy framework
    - **Frontend**: Streamlit interface
    - **Language Models**: [Ollama](https://github.com/ollama/ollama)
    - **Vector Storage**: [Chroma DB](https://github.com/chroma-core/chroma)
    - **Monitoring**: [Arize Phoenix](https://github.com/Arize-ai/phoenix)

    ### Example Queries
    Try asking questions like:
    - "I have young kids and limited time for grooming. Which breed would suit my family?"
    - "List the 5 most popular breeds in the dataset"
    - "Which breeds live the longest on average?"
    - "What breeds are known for being both protective and good with families?"

    ### Info
    [GitHub Repository](https://github.com/yourusername/dog-breed-assistant)
    """
)
