from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = Field(alias="APP_NAME")
    app_version: str = Field(alias="APP_VERSION")
    app_port: int = Field(alias="PORT")
    google_api_key: str = Field(alias="GOOGLE_API_KEY")
    model_code: str = Field(alias="MODEL_CODE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

setting = Settings()