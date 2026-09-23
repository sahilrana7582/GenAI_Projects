from fastapi import FastAPI

from app.routes.routes import router
from app.config.settings import settings
from app.core.logging import configure_logging
from app.exceptions.handlers import app_exception_handler
from app.exceptions.base import AppException

configure_logging()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)


app.add_exception_handler(
    AppException,
    app_exception_handler,
)


@app.get("/health")
async def health_check():
    return {
        "status": "UP",
        "service": settings.app_name,
        "version": settings.app_version,
        "model_code": settings.model_code,
    }


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", port=settings.app_port, reload=True)