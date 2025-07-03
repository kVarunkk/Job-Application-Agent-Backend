from sentence_transformers import SentenceTransformer
from torch import Tensor


resume_embedding_store: dict[str, Tensor ] = {}
resume_text_cache: dict[str, str] = {}
model = SentenceTransformer("all-MiniLM-L6-v2")

