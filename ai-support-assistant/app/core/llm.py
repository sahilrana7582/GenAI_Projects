from langchain_openai import ChatOpenAI
from app.config.settings import settings

model = ChatOpenAI(
    model=settings.model_code,
    api_key=settings.openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)
