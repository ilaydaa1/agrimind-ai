"""Application configuration.

Holds two kinds of settings:
  * machine-specific values loaded from .env (the Foundry Local connection)
  * version-controlled tuning constants that define retrieval and generation
    behaviour — kept in code so every change is reviewable and reproducible.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Paths are resolved from this file, not the process working directory, so the
# app behaves the same regardless of where it is launched from. parents[2] maps
# src/agrimind/config.py -> project root; this holds for our fixed layout and
# both run modes (editable install and `streamlit run` with src on the path).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"

load_dotenv(PROJECT_ROOT / ".env")


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is missing or empty. "
            "Copy .env.example to .env and set it before running the app."
        )
    return value


# --- Machine-specific: Microsoft Foundry Local (from .env) ---
FOUNDRY_BASE_URL = _require_env("FOUNDRY_BASE_URL")
FOUNDRY_MODEL = _require_env("FOUNDRY_MODEL")
FOUNDRY_API_KEY = _require_env("FOUNDRY_API_KEY")

# --- Embedding ---
# Multilingual model: the corpus is Turkish, so an English-centric embedder
# would cap retrieval quality no matter how good the rest of the pipeline is.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# --- Chunking ---
# Size is in tokens (measured with the embedder's own tokenizer), not
# characters: Turkish is agglutinative and the E5 model truncates past 512
# tokens, so character-based sizing would risk silent truncation.
# Overlap is a ratio rather than an absolute count, so it stays proportional if
# the size is retuned — one source of truth for the intended ~15% overlap.
# Both are starting values, tuned later against the golden set.
CHUNK_SIZE_TOKENS = 350
CHUNK_OVERLAP_RATIO = 0.15

# --- Retrieval ---
TOP_K = 4

# Refusal gate threshold on cosine similarity: below this, the query pipeline
# refuses instead of generating.
# This is a PROVISIONAL default. The final value is calibrated on the golden set
# during Day-3 evaluation (the score gap between answerable and unanswerable
# questions), after which this line is updated to the calibrated number.
REFUSAL_SIMILARITY_THRESHOLD = 0.80

# --- Vector store ---
CHROMA_COLLECTION_NAME = "agri_documents"

# --- Generation ---
# Low temperature: in a grounded assistant, output variance is a defect, and a
# near-deterministic model also keeps evaluation reproducible.
GENERATION_TEMPERATURE = 0.1
