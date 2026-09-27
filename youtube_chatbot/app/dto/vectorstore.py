from typing import List

from pydantic import BaseModel, Field


class VectorizeRequest(BaseModel):
    video_id: str = Field(..., min_length=1, description="YouTube video ID these chunks belong to")
    chunks: List[str] = Field(..., min_length=1, description="Transcript chunks to embed and store")


class VectorizeResponse(BaseModel):
    video_id: str = Field(description="YouTube video ID")
    chunk_count: int = Field(description="Number of chunks embedded and stored")
    vector_store_path: str = Field(description="Local path where the FAISS index was persisted")


class RetrieveRequest(BaseModel):
    video_id: str = Field(..., min_length=1, description="YouTube video ID to search within")
    query: str = Field(..., min_length=1, description="User's search query")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of matching chunks to return")


class RetrievedChunk(BaseModel):
    chunk_index: int = Field(description="Index of this chunk within the original transcript")
    text: str = Field(description="Matched chunk text")


class RetrieveResponse(BaseModel):
    video_id: str = Field(description="YouTube video ID")
    query: str = Field(description="The query that was searched")
    results: List[RetrievedChunk] = Field(description="Matching chunks, ordered by relevance")
