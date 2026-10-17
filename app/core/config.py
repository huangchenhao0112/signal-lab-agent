from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    项目配置。

    程序会自动从项目根目录的.env文件读取同名配置。
    """

    app_name: str = "SignalLab Agent"
    app_version: str = "0.1.0"

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_timeout: float = 60.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str = "qwen3.7-text-embedding-flash"
    embedding_dimensions: int = 1024

    chroma_dir: str = "data/chroma"
    chroma_collection: str = "signal_knowledge"
    top_k: int = 3

    redis_url: str = "redis://localhost:6379/0"

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

@lru_cache
def get_settings() -> Settings:
    """
    创建并缓存配置对象。

    第一次调用时读取.env，
    后续调用直接使用同一个Settings对象。
    """

    return Settings()