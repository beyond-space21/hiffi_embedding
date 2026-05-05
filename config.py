from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    RABBITMQ_HOST: str
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str
    RABBITMQ_PASSWORD: str
    RABBITMQ_VHOST: str = "/"
    RABBITMQ_QUEUE: str = "video_embedding"
    RABBITMQ_DLQ: str = "video_embedding_dlq"
    RABBITMQ_PREFETCH_COUNT: int = 1
    WORKER_PROCESSES: int = 1

    QDRANT_URL: str
    COLLECTION_NAME: str = "video_search"

    OPENAI_API_KEY: str
    BASE_API_URL: str | None = None
    AUTH_X_APP: str | None = None
    CACHE_DIR: str = "/mount/disk/huggingface_models"
    TEMP_DIR: str = "temp"
    FRAME_EXTRACT_FPS: float = 1.0
    FRAME_BATCH_SIZE: int = 32
    AUDIO_BATCH_SIZE: int = 16

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
