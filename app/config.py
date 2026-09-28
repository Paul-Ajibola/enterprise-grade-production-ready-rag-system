import os
from dotenv import load_dotenv


load_dotenv()


# set up class for the API keys present
class Settings:
    # ----- GEMINI EMBEDDINGS ----
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    # ----- QDRANT DB ----
    QDRANT_URL = os.getenv('QDRANT_CLUSTER_ENDPOINT')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')    
    QDRANT_COLLECTION = "enterprise_grade_rag"

    # ----- COHERE EMBEDDINGS ----
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')

    # ----- REASONING ENGINE -----
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")


    # ----- LLM GATEWAY -----
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG = "rag"    # primary: @rag/llama-3.3-70b-versatile
    GROQ_SLUG_2 = "brag"   # fallback: @brag/llama-3.1-8b-instant


    # ---- OBSERVABILITY --------
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "enterprise-grade-rag")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://")


#  Apply LangChain environment variables for automatic tracing
os.environ["LANGSMITH_TRACING"] = Settings.LANGSMITH_TRACING
os.environ["LANGSMITH_API_KEY"] = Settings.LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = Settings.LANGSMITH_PROJECT
os.environ["LANGSMITH_ENDPOINT"] = Settings.LANGSMITH_ENDPOINT


# Instantiate the `Settings` class  
settings = Settings()


