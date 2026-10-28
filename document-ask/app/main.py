from dotenv import load_dotenv

load_dotenv()

from ingestion.loader import dir_loader
from ingestion.splitter import text_splitter
from vector.vector import VectorStoreService
from ingestion.retriever import RetrieverService
from langchain_core.prompts import PromptTemplate
from core.llm import model



docs = dir_loader.load()

chunks = text_splitter.split_documents(docs)

print(f"Documents loaded: {len(docs)}")
print(f"Chunks created: {len(chunks)}")

vector_store_service = VectorStoreService()

vector_store = vector_store_service.vectorize_chunks(chunks)

retriever_service = RetrieverService()

query = "how the go programming work and how the thigns work inside that."

results = retriever_service.similarity_search(
    vector_store,
    query,
    k=3
)


context = "\n\n".join(
    result.page_content
    for result in results
)

prompt = PromptTemplate(
    template="""
        You are a helpful software development assistant.

        Answer the user's question using only the provided context.

        If the answer cannot be found in the context, clearly say:
        "I don't have enough information in the provided documents."

        Do not make up information or use knowledge outside the provided context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
    input_variables=["context", "question"]
)

formatted_prompt = prompt.invoke({
    "context": context,
    "question": query
})

response = model.invoke(formatted_prompt)

print("========== CONTENT ==========")
print(response.content)

print("\n========== ADDITIONAL KWARGS ==========")
print(response.additional_kwargs)

print("\n========== REFUSAL ==========")
print(response.additional_kwargs.get("refusal"))

print("\n========== RESPONSE METADATA ==========")
print("Model Provider:", response.response_metadata.get("model_provider"))
print("Model Name:", response.response_metadata.get("model_name"))
print("System Fingerprint:", response.response_metadata.get("system_fingerprint"))
print("Response ID:", response.response_metadata.get("id"))
print("Finish Reason:", response.response_metadata.get("finish_reason"))
print("Logprobs:", response.response_metadata.get("logprobs"))

print("\n========== TOKEN USAGE ==========")
token_usage = response.response_metadata.get("token_usage")

print("Prompt Tokens:", token_usage.get("prompt_tokens"))
print("Completion Tokens:", token_usage.get("completion_tokens"))
print("Total Tokens:", token_usage.get("total_tokens"))

print("\n========== COMPLETION TOKEN DETAILS ==========")
completion_details = token_usage.get("completion_tokens_details")

print("Accepted Prediction Tokens:",
      completion_details.get("accepted_prediction_tokens"))

print("Audio Tokens:",
      completion_details.get("audio_tokens"))

print("Reasoning Tokens:",
      completion_details.get("reasoning_tokens"))

print("Rejected Prediction Tokens:",
      completion_details.get("rejected_prediction_tokens"))

print("Text Tokens:",
      completion_details.get("text_tokens"))

print("\n========== PROMPT TOKEN DETAILS ==========")
prompt_details = token_usage.get("prompt_tokens_details")

print("Audio Tokens:",
      prompt_details.get("audio_tokens"))

print("Cache Write Tokens:",
      prompt_details.get("cache_write_tokens"))

print("Cached Tokens:",
      prompt_details.get("cached_tokens"))

print("Image Tokens:",
      prompt_details.get("image_tokens"))

print("Text Tokens:",
      prompt_details.get("text_tokens"))

print("\n========== MESSAGE INFO ==========")
print("Message ID:", response.id)
print("Tool Calls:", response.tool_calls)
print("Invalid Tool Calls:", response.invalid_tool_calls)

print("\n========== USAGE METADATA ==========")
print("Input Tokens:", response.usage_metadata.get("input_tokens"))
print("Output Tokens:", response.usage_metadata.get("output_tokens"))
print("Total Tokens:", response.usage_metadata.get("total_tokens"))

print("\n========== OUTPUT TOKEN DETAILS ==========")
output_details = response.usage_metadata.get("output_token_details")

print("Reasoning Tokens:",
      output_details.get("reasoning"))

print("\n========== INPUT TOKEN DETAILS ==========")
input_details = response.usage_metadata.get("input_token_details")

print("Audio Tokens:",
      input_details.get("audio"))

print("Cache Read Tokens:",
      input_details.get("cache_read"))

print("Cache Creation Tokens:",
      input_details.get("cache_creation"))

