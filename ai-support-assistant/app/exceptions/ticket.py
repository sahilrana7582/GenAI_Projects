from app.exceptions.base import AppException


class TicketProcessingException(AppException):

    def __init__(self, message: str = "Failed to process ticket"):
        super().__init__(
            message=message,
            status_code=500,
        )