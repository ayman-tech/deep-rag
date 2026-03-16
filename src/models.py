"""
Shared model registry — lazy-loads each model once on first access.
Background pre-loading starts at app startup so models are warm
by the time the user needs them.
"""

import threading
from config import settings

_embed_model = None
_rerank_model = None
_lock = threading.Lock()
_loading_done = threading.Event()


def get_embed_model():
    global _embed_model
    with _lock:
        if _embed_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {settings.DENSE_MODEL}")
            _embed_model = SentenceTransformer(settings.DENSE_MODEL)
    return _embed_model


def get_rerank_model():
    global _rerank_model
    with _lock:
        if _rerank_model is None:
            from sentence_transformers import CrossEncoder
            print(f"Loading rerank model: {settings.RERANK_MODEL}")
            _rerank_model = CrossEncoder(settings.RERANK_MODEL)
    return _rerank_model


def start_background_loading():
    """Kick off model loading in a background thread at app startup."""
    def _load():
        get_embed_model()
        get_rerank_model()
        _loading_done.set()
        print("✅ All models loaded and ready.")

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()


def models_ready() -> bool:
    """Return True if both models have finished loading."""
    return _loading_done.is_set()
