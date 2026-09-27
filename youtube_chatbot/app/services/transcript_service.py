from typing import List

from youtube_transcript_api import CouldNotRetrieveTranscript, YouTubeTranscriptApi

from app.dto.transcript import TranscriptResponse
from app.exceptions.transcript import TranscriptFetchException, TranscriptNotFoundException


class TranscriptService:

    def __init__(self):
        self._api = YouTubeTranscriptApi()

    def get_transcript(self, video_id: str, languages: List[str]) -> TranscriptResponse:
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

        return TranscriptResponse(
            video_id=fetched.video_id,
            language=fetched.language,
            language_code=fetched.language_code,
            is_generated=fetched.is_generated,
            transcript=transcript_text,
        )
