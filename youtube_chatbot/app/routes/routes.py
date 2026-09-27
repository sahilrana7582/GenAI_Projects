from typing import List

from fastapi import APIRouter, Query

from app.dto.qa import AskRequest, AskResponse
from app.dto.transcript import TranscriptChunksResponse, TranscriptResponse
from app.dto.vectorstore import RetrieveRequest, RetrieveResponse, VectorizeRequest, VectorizeResponse
from app.services.qa_service import QAService
from app.services.transcript_service import TranscriptService
from app.services.vector_store_service import VectorStoreService


router = APIRouter(
    prefix="/video",
    tags=["Video"],
)

transcript_service = TranscriptService()
vector_store_service = VectorStoreService()
qa_service = QAService(vector_store_service)


@router.get("/transcript", response_model=TranscriptResponse)
def get_video_transcript(
    video_id: str = Query(..., min_length=1, description="YouTube video ID"),
    languages: List[str] = Query(
        default=["en"],
        description="Preferred transcript languages, in priority order",
    ),
):
    return transcript_service.get_transcript(video_id=video_id, languages=languages)


@router.get("/chunk", response_model=TranscriptChunksResponse)
def get_video_transcript_chunks(
    video_id: str = Query(..., min_length=1, description="YouTube video ID"),
    languages: List[str] = Query(
        default=["en"],
        description="Preferred transcript languages, in priority order",
    ),
):
    return transcript_service.get_transcript_chunks(video_id=video_id, languages=languages)


@router.post("/vectorize", response_model=VectorizeResponse)
def vectorize_video_chunks(request: VectorizeRequest):
    return vector_store_service.vectorize_chunks(
        video_id=request.video_id, chunks=request.chunks
    )


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_video_chunks(request: RetrieveRequest):
    return vector_store_service.retrieve(
        video_id=request.video_id, query=request.query, top_k=request.top_k
    )


@router.post("/ask", response_model=AskResponse)
def ask_video_question(request: AskRequest):
    return qa_service.ask(video_id=request.video_id, query=request.query)
