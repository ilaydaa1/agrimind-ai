"""Application-layer RAG pipeline: a question in, a grounded answer or refusal out."""

from dataclasses import dataclass

from agrimind.config import REFUSAL_SIMILARITY_THRESHOLD
from agrimind.infrastructure.embedding.e5_embedder import E5Embedder
from agrimind.infrastructure.llm.foundry_local_client import FoundryLocalClient
from agrimind.infrastructure.vectorstore.chroma_store import ChromaStore, RetrievedChunk

_REFUSAL_ANSWER = (
    "Yüklenen belgelerde bu soruyu yanıtlamak için yeterli bilgi bulunmamaktadır."
)

_PROMPT_TEMPLATE = """Sen bir tarımsal belge asistanısın. Soruyu YALNIZCA aşağıdaki \
bağlamı kullanarak yanıtla. Bağlam soruyu yanıtlamak için yeterli bilgi içermiyorsa, \
bunu açıkça belirt ve hiçbir bilgi uydurma. Yanıtı yalnızca Türkçe yaz.

Bağlam
--------
{context}

Soru
--------
{question}"""


@dataclass(frozen=True, slots=True)
class Citation:
    document_id: str
    page_number: int


@dataclass(frozen=True, slots=True)
class RAGResponse:
    answer: str
    citations: list[Citation]
    max_similarity: float


class RAGPipeline:
    """Orchestrates embedding, retrieval, the refusal gate, generation, and
    citations. Owns the sequence only; every operation is delegated to an
    injected adapter.
    """

    def __init__(
        self,
        embedder: E5Embedder,
        store: ChromaStore,
        llm: FoundryLocalClient,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm

    def answer(self, question: str) -> RAGResponse:
        query_embedding = self._embedder.embed(question)
        chunks = self._store.retrieve(query_embedding)
        max_similarity = max((chunk.similarity for chunk in chunks), default=0.0)

        if max_similarity < REFUSAL_SIMILARITY_THRESHOLD:
            return RAGResponse(
                answer=_REFUSAL_ANSWER,
                citations=[],
                max_similarity=max_similarity,
            )

        answer = self._llm.complete(_build_prompt(question, chunks))
        citations = [
            Citation(document_id=chunk.document_id, page_number=chunk.page_number)
            for chunk in chunks
        ]
        return RAGResponse(
            answer=answer,
            citations=citations,
            max_similarity=max_similarity,
        )


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(chunk.text for chunk in chunks)
    return _PROMPT_TEMPLATE.format(context=context, question=question)
