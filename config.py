from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DEEPSEEK_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    COLLECTION_NAME: str = "gold_standard_rag"
    
    # Models
    DENSE_MODEL: str = "BAAI/bge-m3"
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()