from typing import List

from pydantic import BaseModel, Field


class VectorizeRequest(BaseModel):
    video_id: str = Field(..., min_length=1, description="YouTube video ID these chunks belong to")
    chunks: List[str] = Field(..., min_length=1, description="Transcript chunks to embed and store")


class VectorizeResponse(BaseModel):
    video_id: str = Field(description="YouTube video ID")
    chunk_count: int = Field(description="Number of chunks embedded and stored")
    vector_store_path: str = Field(description="Local path where the FAISS index was persisted")
