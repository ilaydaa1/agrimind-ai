"""Embedding generation with the multilingual E5 model."""

from sentence_transformers import SentenceTransformer

from agrimind.config import EMBEDDING_MODEL
from agrimind.infrastructure.chunking.recursive_chunker import Chunk

# E5 models are trained with these input prefixes and underperform without them:
# queries are prefixed "query: " and stored passages "passage: ".
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


class E5Embedder:
    """Loads intfloat/multilingual-e5-base once and produces normalized
    embeddings. The model's own tokenizer backs `count_tokens`, so chunking and
    embedding measure length identically without loading the tokenizer twice.
    """

    def __init__(self) -> None:
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def embed(self, text: str) -> list[float]:
        """Embed a single query string (applies the E5 query prefix)."""
        embedding = self._model.encode(_QUERY_PREFIX + text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed chunk texts as passages in one batched call."""
        passages = [_PASSAGE_PREFIX + chunk.text for chunk in chunks]
        embeddings = self._model.encode(passages, normalize_embeddings=True)
        return embeddings.tolist()
