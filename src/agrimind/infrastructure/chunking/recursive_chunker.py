"""Token-aware, boundary-preserving chunking of extracted PDF pages."""

from collections.abc import Callable
from dataclasses import dataclass

from agrimind.config import CHUNK_OVERLAP_RATIO, CHUNK_SIZE_TOKENS
from agrimind.infrastructure.extraction.pymupdf_extractor import ExtractedPage

# Boundaries from coarsest to finest: paragraph, line, sentence, word, character.
# The empty string (character split) is the final fallback, so the recursion
# always terminates and no fragment is left larger than the token budget.
_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    page_number: int


def chunk_pages(
    pages: list[ExtractedPage],
    count_tokens: Callable[[str], int],
) -> list[Chunk]:
    """Split each page into overlapping, token-bounded chunks.

    Chunks never span pages, so each one keeps a single source page number;
    empty pages are skipped without renumbering the others. `count_tokens` must
    measure length with the embedding model's tokenizer
    (intfloat/multilingual-e5-base) so chunk sizes match what is embedded.

    Size is a target: a chunk may exceed it by at most the overlap plus one
    fragment, which stays well under the model's 512-token limit.
    """
    size = CHUNK_SIZE_TOKENS
    overlap = round(CHUNK_SIZE_TOKENS * CHUNK_OVERLAP_RATIO)
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text:
            continue
        pieces = _split_to_pieces(page.text, _SEPARATORS, size, count_tokens)
        chunks.extend(
            Chunk(text=text, page_number=page.page_number)
            for text in _pack(pieces, size, overlap, count_tokens)
        )
    return chunks


def _split_to_pieces(
    text: str,
    separators: tuple[str, ...],
    size: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """Break text into fragments that each fit the token budget, preferring the
    coarsest boundary that works before falling back to finer ones."""
    if not separators or count_tokens(text) <= size:
        return [text]
    separator, finer = separators[0], separators[1:]
    pieces: list[str] = []
    for fragment in _split_keep(text, separator):
        if not fragment:
            continue
        if count_tokens(fragment) <= size:
            pieces.append(fragment)
        else:
            pieces.extend(_split_to_pieces(fragment, finer, size, count_tokens))
    return pieces


def _split_keep(text: str, separator: str) -> list[str]:
    """Split on separator while keeping it attached to the preceding fragment,
    so concatenating the fragments reproduces the original text."""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    return [part + separator for part in parts[:-1]] + parts[-1:]


def _pack(
    pieces: list[str],
    size: int,
    overlap: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """Greedily fill chunks up to the token budget, carrying an overlapping tail
    of trailing fragments into the next chunk."""
    counted = [(piece, count_tokens(piece)) for piece in pieces]
    chunks: list[str] = []
    window: list[tuple[str, int]] = []
    window_tokens = 0
    for piece, tokens in counted:
        if window and window_tokens + tokens > size:
            chunks.append("".join(text for text, _ in window))
            window = _overlap_tail(window, overlap)
            window_tokens = sum(count for _, count in window)
        window.append((piece, tokens))
        window_tokens += tokens
    if window:
        chunks.append("".join(text for text, _ in window))
    return chunks


def _overlap_tail(
    window: list[tuple[str, int]], overlap: int
) -> list[tuple[str, int]]:
    """Return the trailing fragments whose combined length fits the overlap
    budget; empty when even the last fragment alone exceeds it."""
    tail: list[tuple[str, int]] = []
    tail_tokens = 0
    for piece, tokens in reversed(window):
        if tail_tokens + tokens > overlap:
            break
        tail.insert(0, (piece, tokens))
        tail_tokens += tokens
    return tail
