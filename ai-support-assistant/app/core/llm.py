from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import settings

model = ChatGoogleGenerativeAI(
    model = settings.model_code,
    google_api_key=settings.google_api_key,
    temperature=0,
)