from pydantic_settings import SettingsConfigDict , BaseSettings 
from pathlib import Path

PROJECT_ROOT = Path(__name__).resolve().parents[0]

class Settings(BaseSettings):
    
    MODEL_NAME : str = "BiLSTM"
    
    HIDDEN_SIZE : int
    EMBEDDING_DIM : int
    DROPOUT : int
    BATCH_SIZE : int
    EPOCHS : int
    LR : float
    
    
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )