from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    port: int = 7860
    embedding_dim: int = 192
    num_threads: int = 2
    vad_threshold: float = 0.5
    cors_origins: str = "*"
    debug: bool = False
    models_dir: str = "onnx_models"
    default_room_code: str = "DEFAULT"
    room_idle_ttl_seconds: float = 7200.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
