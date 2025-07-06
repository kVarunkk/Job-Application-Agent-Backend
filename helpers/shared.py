from sentence_transformers import SentenceTransformer
from typing import Any

resume_text_cache: dict[str, str] = {}
model = SentenceTransformer("all-MiniLM-L6-v2")
job_embedding_store: dict[str, Any] = {}
