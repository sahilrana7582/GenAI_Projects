from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI, HTTPException, status
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
import os 

# Load .env
load_dotenv()

# ============================================================ 
# # Logging # 
# ============================================================ 
logging.basicConfig(
     level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", 
)
logger = logging.getLogger(__name__)

#App Model
class Settings: 
    APP_NAME = "Prompt Service" 
    APP_VERSION = "1.0.0" 
    HOST = "0.0.0.0" 
    PORT = 8888

settings = Settings()

# Model
class UserPromptRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}

    prompt_text: str = Field(
        min_length=1,
        max_length=4000,
        description="Question or prompt provided by the user",
        examples=["Explain how binary search works"],
    )

# ============================================================
# # LLM Response Model #
#  ============================================================
class LLMAnswer(BaseModel):
    answer: List[str] = Field(description="The answer to the user's question with the proper bullet points. Bullet Points Must Be atleast 10 it can be more but 10 is the minimum value")
    summary: str = Field(description="A concise summary of the answer")


class UserPromptResponse(LLMAnswer):
    question: str = Field(description="The original question that was asked")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is a REST API?",
                "answer": [
                    "REST is an architectural style for designing networked applications.",
                    "It uses standard HTTP methods such as GET, POST, PUT, and DELETE.",
                ],
                "summary": "REST APIs expose resources over HTTP using stateless, standard methods.",
            }
        }
    }

# LLM Model
mode_code=os.getenv("MODEL_CODE")
model = ChatGoogleGenerativeAI(
    model=mode_code
)

structured_model = model.with_structured_output(LLMAnswer)

prompt = PromptTemplate( 
    input_variables=[ "question"], 
    template=""" You are an expert AI assistant with broad knowledge across software engineering, computer science, mathematics, science, business, finance, history, and general knowledge. Your task is to provide accurate, useful, and easy-to-understand answers to the user's question. Follow these rules: 1. Understand the user's question carefully before answering. 2. Give a direct answer first, then provide additional explanation when it is useful. 3. Adapt the depth of the explanation to the complexity of the question. 4. When explaining a technical concept, prefer: - A simple explanation first. - A practical example. - Code examples when they are relevant. - Important edge cases or limitations when applicable. 5. When the question involves multiple concepts, structure the answer logically and explain the relationship between them. 6. Do not invent facts, sources, APIs, code behavior, or other information. If you are uncertain, clearly state the uncertainty. 7. Distinguish between established facts, assumptions, and recommendations when relevant. 8. If the user's question is ambiguous and the ambiguity materially affects the answer, clearly state the assumption you are making. 9. Keep the response focused on the user's question. Avoid unnecessary information. 10. Use clear formatting such as headings, bullet points, numbered steps, tables, or code blocks when they improve readability. 11. Return the response strictly according to the following output User Question: {question} """,
)


chain = prompt | structured_model


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION
)

@app.get("/health") 
async def health_check(): 
    return { 
        "status": "UP", 
        "service": settings.APP_NAME, 
        "version": settings.APP_VERSION, 
    }


@app.post(
    "/prompt",
    response_model=UserPromptResponse
)
async def process_prompt(request: UserPromptRequest):
    logger.info("Processing user request")

    try:
        result = chain.invoke({"question": request.prompt_text})
    except Exception as exc:
        logger.error(f"Model call failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to get a response from the model. Please try again.",
        )

    return UserPromptResponse(
        question=request.prompt_text,
        answer=result.answer,
        summary=result.summary,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("APP_PORT", 8888)),
        reload=True,
    )