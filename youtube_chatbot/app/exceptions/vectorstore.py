from app.exceptions.base import AppException


class VectorStoreException(AppException):

    def __init__(self, message: str = "Failed to embed and store chunks"):
        super().__init__(
            message=message,
            status_code=500,
        )


class VectorStoreNotFoundException(AppException):

    def __init__(self, message: str = "No vector store found for this video"):
        super().__init__(
            message=message,
            status_code=404,
        )
