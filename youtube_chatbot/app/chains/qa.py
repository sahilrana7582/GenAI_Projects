from langchain_core.output_parsers import StrOutputParser

from app.core.llm import chat_model
from app.prompts.qa import qa_prompt

qa_chain = qa_prompt | chat_model | StrOutputParser()
