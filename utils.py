import platform
import signal
import threading
from contextlib import contextmanager
from typing import Generator
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import pickle

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds: int) -> Generator:
    """Cross-platform timeout context manager."""
    def timeout_handler():
        raise TimeoutException("Operation timed out")
        
    if platform.system() != 'Windows':
        signal.signal(signal.SIGALRM, lambda *_: TimeoutException("Operation timed out"))
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        timer = threading.Timer(seconds, timeout_handler)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

def get_or_create_embeddings(persist_directory: str, model_name: str):
    """Create or load cached embeddings."""
    cache_file = Path(persist_directory) / f"{model_name.replace('/', '_').replace('-', '_')}_embeddings.pkl"
    os.makedirs(persist_directory, exist_ok=True)
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                embeddings = SentenceTransformerEmbeddings(model_name)
                embeddings.model.state_dict = pickle.load(f)
                return embeddings
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}")
    
    embeddings = SentenceTransformerEmbeddings(model_name)
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings.model.state_dict(), f)
    
    return embeddings