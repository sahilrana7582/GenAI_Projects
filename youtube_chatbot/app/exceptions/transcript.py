from app.exceptions.base import AppException


class TranscriptNotFoundException(AppException):

    def __init__(self, message: str = "No transcript found for this video"):
        super().__init__(
            message=message,
            status_code=404,
        )


class TranscriptFetchException(AppException):

    def __init__(self, message: str = "Failed to fetch transcript"):
        super().__init__(
            message=message,
            status_code=502,
        )
