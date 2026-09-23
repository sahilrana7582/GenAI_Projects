import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from app.exceptions.base import AppException

logger = logging.getLogger(__name__)


async def app_exception_handler(
    request: Request,
    exc: AppException,
):
    logger.exception("Request to %s failed: %s", request.url.path, exc.message)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
        },
    )