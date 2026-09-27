from typing import List

from fastapi import APIRouter, Query

from app.dto.transcript import TranscriptResponse
from app.services.transcript_service import TranscriptService


router = APIRouter(
    prefix="/video",
    tags=["Video"],
)

transcript_service = TranscriptService()


@router.get("/transcript", response_model=TranscriptResponse)
def get_video_transcript(
    video_id: str = Query(..., min_length=1, description="YouTube video ID"),
    languages: List[str] = Query(
        default=["en"],
        description="Preferred transcript languages, in priority order",
    ),
):
    return transcript_service.get_transcript(video_id=video_id, languages=languages)
