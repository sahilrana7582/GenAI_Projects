from app.exceptions.base import AppException


class VectorStoreException(AppException):

    def __init__(self, message: str = "Failed to embed and store chunks"):
        super().__init__(
            message=message,
            status_code=500,
        )
