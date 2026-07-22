"""Composition root: construct the long-lived adapters once and wire the pipelines."""

from dataclasses import dataclass

from agrimind.application.ingestion_pipeline import IngestionPipeline
from agrimind.application.rag_pipeline import RAGPipeline
from agrimind.infrastructure.embedding.e5_embedder import E5Embedder
from agrimind.infrastructure.llm.foundry_local_client import FoundryLocalClient
from agrimind.infrastructure.vectorstore.chroma_store import ChromaStore


@dataclass(frozen=True, slots=True)
class Application:
    ingestion: IngestionPipeline
    rag: RAGPipeline


def build_application() -> Application:
    embedder = E5Embedder()
    store = ChromaStore()
    llm = FoundryLocalClient()
    return Application(
        ingestion=IngestionPipeline(embedder=embedder, store=store),
        rag=RAGPipeline(embedder=embedder, store=store, llm=llm),
    )
