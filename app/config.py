import os
from dotenv import load_dotenv


load_dotenv()


# set up class for the API keys present
class Settings:
    # ----- QDRANT DB ----
    QDRANT_URL = os.getenv('QDRANT_CLUSTER_ENDPOINT')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')    
    QDRANT_COLLECTION = "enterprise_grade_rag"
    CHECKPOINT_DB_PATH = os.getenv(
        "CHECKPOINT_DB_PATH", "~/.enterprise-rag/checkpoints.sqlite"
    )

    # ----- COHERE EMBEDDINGS ----
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')

    # ----- REASONING ENGINE -----
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = "llama-3.3-70b-versatile"
    # ----- LLM GATEWAY -----
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG = os.getenv("PORTKEY_PRIMARY_SLUG") or "rag"
    GROQ_SLUG_2 = os.getenv("PORTKEY_FALLBACK_SLUG") or "brag"


    # ---- OBSERVABILITY --------
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or None
    # Tracing is optional. Enable it by default only when credentials exist.
    LANGSMITH_TRACING = os.getenv(
        "LANGSMITH_TRACING",
        "true" if LANGSMITH_API_KEY else "false",
    )
    if not LANGSMITH_API_KEY:
        LANGSMITH_TRACING = "false"
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "enterprise-grade-rag")
    LANGSMITH_ENDPOINT = (
        os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
    )


# Apply LangChain environment variables for automatic tracing.
os.environ["LANGSMITH_TRACING"] = Settings.LANGSMITH_TRACING
if Settings.LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = Settings.LANGSMITH_API_KEY
else:
    os.environ.pop("LANGSMITH_API_KEY", None)
os.environ["LANGSMITH_PROJECT"] = Settings.LANGSMITH_PROJECT
os.environ["LANGSMITH_ENDPOINT"] = Settings.LANGSMITH_ENDPOINT


# Instantiate the `Settings` class  
settings = Settings()


