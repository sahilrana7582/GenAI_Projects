import os

from langchain_openai import ChatOpenAI


model = ChatOpenAI(
    model=os.getenv("MODEL_NAME")
)