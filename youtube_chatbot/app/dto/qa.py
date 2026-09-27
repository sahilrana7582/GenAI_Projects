from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    video_id: str = Field(..., min_length=1, description="YouTube video ID to ask about")
    query: str = Field(..., min_length=1, description="User's question about the video")


class AskResponse(BaseModel):
    video_id: str = Field(description="YouTube video ID")
    query: str = Field(description="The question that was asked")
    answer: str = Field(description="The LLM's answer, grounded in retrieved transcript chunks")
