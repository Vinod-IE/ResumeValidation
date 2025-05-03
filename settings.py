# settings.py
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class Settings:
    FILES_FOLDER = os.getenv('FILES_FOLDER', './data')
    GROQ_KEY = os.getenv('GROQ_KEY')
    GROQ_MODEL = os.getenv('GROQ_MODEL')
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME')
    FAST_EMBDED_MODEL = os.getenv("FAST_EMBDED_MODEL")
    LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
    OPENAI_EMBEDDINGS_MODEL = os.getenv('OPENAI_EMBEDDINGS_MODEL')
    SENTENCE_TRANSFORMER_MODEL = os.getenv('SENTENCE_TRANSFORMER_MODEL')

    @classmethod
    def check(cls):
        required = ['GROQ_KEY', 'GROQ_MODEL']  # Removed OPENAI_API_KEY
        for var in required:
            if not getattr(cls, var):
                raise ValueError(f"Missing {var} in environment")

settings = Settings()
settings.check()

