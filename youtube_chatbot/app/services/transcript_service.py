from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    FetchedTranscript,
    YouTubeTranscriptApi,
)

from app.dto.transcript import TranscriptChunksResponse, TranscriptResponse
from app.exceptions.transcript import TranscriptFetchException, TranscriptNotFoundException


class TranscriptService:

    def __init__(self):
        self._api = YouTubeTranscriptApi()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    def get_transcript(self, video_id: str, languages: List[str]) -> TranscriptResponse:
        fetched, transcript_text = self._fetch_transcript(video_id, languages)
        chunks = self.chunk_transcript(transcript_text)

        return TranscriptResponse(
            video_id=fetched.video_id,
            language=fetched.language,
            language_code=fetched.language_code,
            is_generated=fetched.is_generated,
            transcript=transcript_text,
            chunk_count=len(chunks),
        )

    def get_transcript_chunks(
        self, video_id: str, languages: List[str]
    ) -> TranscriptChunksResponse:
        _, transcript_text = self._fetch_transcript(video_id, languages)
        chunks = self.chunk_transcript(transcript_text)

        return TranscriptChunksResponse(
            video_id=video_id,
            chunk_count=len(chunks),
            chunks=chunks,
        )

    def chunk_transcript(self, transcript_text: str) -> List[str]:
        return self._splitter.split_text(transcript_text)

    def _fetch_transcript(
        self, video_id: str, languages: List[str]
    ) -> tuple[FetchedTranscript, str]:
        try:
            fetched = self._api.fetch(video_id=video_id, languages=languages)
        except CouldNotRetrieveTranscript as exc:
            raise TranscriptNotFoundException(
                message=f"No transcript available for video '{video_id}' in languages {languages}"
            ) from exc
        except Exception as exc:
            raise TranscriptFetchException(
                message=f"Failed to fetch transcript for video '{video_id}'"
            ) from exc

        transcript_text = " ".join(snippet.text for snippet in fetched)
        return fetched, transcript_text
