from pathlib import Path
from typing import Union
from pydantic import BaseModel, Field, validator, HttpUrl

class Config(BaseModel):
    """Application configuration for the RAG system."""
    # Data paths
    pdf_dir: Path = Field(
        default=Path("data/tech"), 
        description="Directory containing PDF files"
    )
    persist_directory: str = Field(
        default="embeddings_cache", 
        description="Directory to cache embeddings"
    )
    
    # Model configurations
    embeddings_model_name: str = Field(default="all-MiniLM-L12-v2")
    llm_model_path: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",  # Changed to Qwen model ID
        description="HuggingFace model ID for Qwen"
    )
    vllm_api_url: HttpUrl = Field(
        default="http://localhost:8000/v1"
    )
    
    # RAG parameters
    chunk_size: int = Field(
        default=500,  # Reduced from 900
        ge=100, 
        le=2000,
        description="Text chunk size for splitting"
    )
    chunk_overlap: int = Field(
        default=50, 
        ge=0, 
        le=900,
        description="Overlap between chunks"
    )
    max_input_tokens: int = Field(
        default=1536,
        ge=1,
        le=8192,
        description="Maximum input tokens allowed"
    )
    
    # Generation parameters
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    top_p: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0,
        description="Top-p sampling parameter"
    )
    response_timeout: int = Field(
        default=300,
        ge=1,
        description="Timeout for model response in seconds"
    )
    
    # Search parameters
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for document retrieval"
    )
    fetch_k: int = Field(
        default=10,
        ge=1,
        description="Number of documents to fetch before MMR"
    )
    return_k: int = Field(
        default=3,
        ge=1,
        description="Number of documents to return after reranking"
    )
    
    model_config = {
        "arbitrary_types_allowed": True
    }

    @validator('pdf_dir')
    def validate_paths(cls, v: Path) -> Path:
        """Validate that paths exist."""
        if not v.exists():
            raise ValueError(f"Path {v} does not exist")
        return v

    @validator('chunk_overlap')
    def validate_overlap(cls, v: int, values: dict) -> int:
        """Ensure chunk_overlap is less than chunk_size."""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
