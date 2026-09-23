from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Plain app info - safe to default, nothing breaks if .env leaves these out.
    app_name: str = Field(default="AI Support Assistant", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    app_port: int = Field(default=8000, alias="PORT", ge=1, le=65535)

    # Required config - no default, so the app fails immediately (not mid-request)
    # if these are missing from .env.
    openrouter_api_key: str = Field(alias="OPENROUTER_API_KEY", min_length=1)
    model_code: str = Field(alias="MODEL_CODE", min_length=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
