import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

CHAT_MODEL_CODE = os.getenv("MODEL_CODE", "google/gemma-4-31b-it:free")
EMBEDDING_MODEL_CODE = os.getenv(
    "EMBEDDING_MODEL_CODE", "sentence-transformers/all-MiniLM-L6-v2"
)

# OpenRouter's free tier shares a rate-limited pool per model/provider. If
# CHAT_MODEL_CODE gets a 429, OpenRouter automatically retries the next model
# in this list instead of failing the request outright.
# Note: OpenRouter caps this array at 3 models total.
CHAT_FALLBACK_MODELS = [
    CHAT_MODEL_CODE,
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3.5-lightning:free",
]

chat_model = ChatOpenAI(
    model=CHAT_MODEL_CODE,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    extra_body={"models": CHAT_FALLBACK_MODELS},
)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_CODE)
