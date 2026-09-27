from langchain_core.prompts import PromptTemplate

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant answering questions about a YouTube video using only the transcript excerpts provided below.

Transcript excerpts:
{context}

Question: {question}

Instructions:
- Answer using only the information in the transcript excerpts above.
- If the excerpts don't contain enough information to answer, say so clearly instead of guessing.
- Be concise and direct.

Answer:""",
)
