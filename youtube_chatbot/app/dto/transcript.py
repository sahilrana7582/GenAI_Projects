from pydantic import BaseModel, Field


class TranscriptResponse(BaseModel):
    video_id: str = Field(description="YouTube video ID")
    language: str = Field(description="Full name of the transcript language")
    language_code: str = Field(description="Language code of the transcript, e.g. 'en'")
    is_generated: bool = Field(description="Whether the transcript was auto-generated")
    transcript: str = Field(description="Full transcript text")
