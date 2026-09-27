from app.exceptions.base import AppException


class AnswerGenerationException(AppException):

    def __init__(self, message: str = "Failed to generate an answer"):
        super().__init__(
            message=message,
            status_code=502,
        )
