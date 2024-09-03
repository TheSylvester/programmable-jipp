import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    openai_api_key: str = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = Field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    groq_api_key: str = Field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY", "")
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
