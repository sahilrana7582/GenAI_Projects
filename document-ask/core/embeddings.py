from langchain_openai import OpenAIEmbeddings
import os

model_name =os.getenv("EMBEDDING_MODEL")

embeddings = OpenAIEmbeddings(
    model=model_name
)