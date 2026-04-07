"""
Document chunking strategies for biomedical RAG.

Three complementary chunking strategies:

1. SentenceChunker — splits on sentence boundaries with configurable overlap.
   Best for PubMed abstracts and short clinical note sections.

2. SemanticChunker — detects topic boundaries using embedding cosine similarity.
   Groups semantically coherent sentences; splits at high-dissimilarity gaps.
   Best for long clinical notes and full-text articles.

3. SectionAwareChunker — respects pre-identified clinical note sections,
   chunking within sections and preserving section metadata.
   Best for structured MIMIC-III discharge summaries.

All chunkers produce a standardized chunk schema compatible with the vector
store and retrieval modules.

References:
  - Greg Kamradt's semantic chunking approach:
    https://github.com/FullStackRetrieval-com/RetrievalTutorials
  - LangChain SemanticChunker implementation
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A single text chunk ready for embedding and indexing.

    Attributes:
        chunk_id: Unique identifier (format: <doc_id>_chunk_<n>).
        text: Chunk text content.
        metadata: Document-level metadata plus chunk-level additions.
        token_count: Approximate token count (word-based estimate).
        char_start: Character offset in source document.
        char_end: Character end offset in source document.
    """

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0

    def __post_init__(self) -> None:
        if not self.token_count:
            # Approximate token count: words * 1.3 (accounts for subword tokenization)
            self.token_count = int(len(self.text.split()) * 1.3)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

class BiomedicalSentenceSplitter:
    """
    Sentence splitter aware of biomedical abbreviations.

    Standard sentence splitters (spaCy, NLTK) struggle with biomedical text:
      - "t.i.d." (three times daily) — internal periods
      - "Dr. Smith" — title abbreviations
      - "e.g." / "i.e." — Latin abbreviations
      - "Fig." / "et al." — citation abbreviations
      - Lab values: "Na+ 140 mEq/L" — not sentence ends

    This splitter uses a heuristic approach with a biomedical abbreviation
    list to avoid splitting on non-sentence boundaries.
    """

    # Common biomedical abbreviations that should NOT trigger sentence splits
    BIOMEDICAL_ABBREVS = {
        # Dosing
        "q.d.", "q.i.d.", "t.i.d.", "b.i.d.", "p.r.n.", "p.o.", "s.l.",
        "i.v.", "i.m.", "s.c.", "i.t.", "i.a.", "i.p.",
        # Titles and common
        "dr.", "mr.", "mrs.", "ms.", "prof.", "jr.", "sr.",
        # Academic/citation
        "fig.", "et al.", "e.g.", "i.e.", "cf.", "vs.",
        # Medical
        "resp.", "temp.", "approx.", "incl.", "excl.", "max.", "min.",
        "dept.", "div.", "no.", "vol.", "pt.", "pts.", "wt.", "ht.",
        # Units
        "mg.", "ml.", "dl.", "kg.", "cm.", "mm.", "hr.", "min.",
    }

    def split(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Clinical or biomedical text.

        Returns:
            List of sentence strings.
        """
        # Protect abbreviations from being split
        protected = text
        abbrev_map: Dict[str, str] = {}
        for i, abbrev in enumerate(self.BIOMEDICAL_ABBREVS):
            token = f"ABBREV{i}ABBREV"
            protected = re.sub(re.escape(abbrev), token, protected, flags=re.IGNORECASE)
            abbrev_map[token] = abbrev

        # Split on sentence-ending punctuation
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)

        # Restore abbreviations
        sentences = []
        for sent in raw_sentences:
            for token, abbrev in abbrev_map.items():
                sent = sent.replace(token, abbrev)
            sentences.append(sent.strip())

        return [s for s in sentences if s]


# ---------------------------------------------------------------------------
# Base chunker
# ---------------------------------------------------------------------------

class BaseChunker(ABC):
    """Abstract base for all chunkers."""

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size  # tokens
        self.overlap = overlap  # tokens of overlap
        self.min_chunk_size = min_chunk_size

    @abstractmethod
    def chunk(self, document: dict) -> List[Chunk]:
        """
        Chunk a RAG document into pieces.

        Args:
            document: Dict with keys 'id', 'text', 'metadata'.

        Returns:
            List of Chunk objects.
        """

    def _make_chunk(
        self,
        doc_id: str,
        text: str,
        metadata: dict,
        chunk_index: int,
        char_start: int = 0,
        char_end: int = 0,
        extra_metadata: Optional[dict] = None,
    ) -> Chunk:
        """Create a Chunk with standard metadata augmentation."""
        chunk_metadata = {
            **metadata,
            "chunk_index": chunk_index,
            "doc_id": doc_id,
        }
        if extra_metadata:
            chunk_metadata.update(extra_metadata)
        return Chunk(
            chunk_id=f"{doc_id}_chunk_{chunk_index}",
            text=text.strip(),
            metadata=chunk_metadata,
            char_start=char_start,
            char_end=char_end if char_end else char_start + len(text),
        )

    @staticmethod
    def _token_count(text: str) -> int:
        """Fast approximate token count (words * 1.3 subword factor)."""
        return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Sentence chunker
# ---------------------------------------------------------------------------

class SentenceChunker(BaseChunker):
    """
    Chunk documents by grouping sentences up to a token budget.

    Consecutive chunks share ``overlap`` tokens of context. This simple
    strategy works well for PubMed abstracts and short sections.

    Args:
        chunk_size: Target token count per chunk (approximate).
        overlap: Tokens of overlap between consecutive chunks.
        min_chunk_size: Minimum tokens; smaller chunks are merged with neighbors.

    Example::

        chunker = SentenceChunker(chunk_size=512, overlap=64)
        chunks = chunker.chunk({"id": "pmid_12345", "text": abstract, "metadata": {}})
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 50,
    ):
        super().__init__(chunk_size, overlap, min_chunk_size)
        self._splitter = BiomedicalSentenceSplitter()

    def chunk(self, document: dict) -> List[Chunk]:
        doc_id = document["id"]
        text = document["text"]
        metadata = document.get("metadata", {})

        sentences = self._splitter.split(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sents: List[str] = []
        current_tokens = 0
        overlap_sents: List[str] = []  # Rolling window for overlap
        chunk_index = 0
        char_pos = 0

        for sent in sentences:
            sent_tokens = self._token_count(sent)

            if current_tokens + sent_tokens > self.chunk_size and current_sents:
                # Emit current chunk
                chunk_text = " ".join(current_sents)
                chunk_start = text.find(current_sents[0], char_pos)
                chunks.append(
                    self._make_chunk(
                        doc_id=doc_id,
                        text=chunk_text,
                        metadata=metadata,
                        chunk_index=chunk_index,
                        char_start=max(0, chunk_start),
                    )
                )
                chunk_index += 1

                # Compute overlap: take sentences from the end of current chunk
                overlap_sents = []
                overlap_tokens = 0
                for prev_sent in reversed(current_sents):
                    t = self._token_count(prev_sent)
                    if overlap_tokens + t > self.overlap:
                        break
                    overlap_sents.insert(0, prev_sent)
                    overlap_tokens += t

                current_sents = overlap_sents + [sent]
                current_tokens = overlap_tokens + sent_tokens
            else:
                current_sents.append(sent)
                current_tokens += sent_tokens

        # Emit remaining sentences
        if current_sents:
            chunk_text = " ".join(current_sents)
            if self._token_count(chunk_text) >= self.min_chunk_size:
                chunk_start = text.rfind(current_sents[0])
                chunks.append(
                    self._make_chunk(
                        doc_id=doc_id,
                        text=chunk_text,
                        metadata=metadata,
                        chunk_index=chunk_index,
                        char_start=max(0, chunk_start),
                    )
                )

        logger.debug("SentenceChunker: %s → %d chunks", doc_id, len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# Semantic chunker
# ---------------------------------------------------------------------------

class SemanticChunker(BaseChunker):
    """
    Chunk documents by detecting semantic topic boundaries.

    Algorithm:
      1. Split text into sentences.
      2. Embed each sentence using the provided embedding function.
      3. Compute cosine similarity between adjacent sentence embeddings.
      4. Find breakpoints where similarity drops below a threshold
         (or is below the (1 - percentile) of all similarities).
      5. Group sentences into chunks at breakpoints.

    This produces topically coherent chunks that respect semantic boundaries,
    which is especially important for long clinical notes where topics shift
    (e.g., from History to Physical Exam to Assessment).

    Args:
        embed_fn: Callable that takes List[str] → np.ndarray (N, D).
                  If None, falls back to SentenceChunker.
        breakpoint_percentile: Percentile of similarity distribution to use
                               as breakpoint threshold. Higher = more chunks.
        chunk_size: Maximum tokens per chunk (fallback splitting).

    Example::

        from src.embeddings.biomedical_embedder import BiomedicalEmbedder
        embedder = BiomedicalEmbedder()
        chunker = SemanticChunker(embed_fn=embedder.encode)
        chunks = chunker.chunk(document)
    """

    def __init__(
        self,
        embed_fn=None,
        breakpoint_percentile: float = 85.0,
        chunk_size: int = 512,
        min_chunk_size: int = 50,
    ):
        super().__init__(chunk_size=chunk_size, min_chunk_size=min_chunk_size)
        self.embed_fn = embed_fn
        self.breakpoint_percentile = breakpoint_percentile
        self._splitter = BiomedicalSentenceSplitter()
        self._fallback = SentenceChunker(chunk_size=chunk_size, min_chunk_size=min_chunk_size)

    def chunk(self, document: dict) -> List[Chunk]:
        if self.embed_fn is None:
            logger.warning(
                "SemanticChunker: no embed_fn provided. Falling back to SentenceChunker."
            )
            return self._fallback.chunk(document)

        doc_id = document["id"]
        text = document["text"]
        metadata = document.get("metadata", {})

        sentences = self._splitter.split(text)
        if len(sentences) < 3:
            # Too short for semantic chunking
            return self._fallback.chunk(document)

        # Embed all sentences
        embeddings = self.embed_fn(sentences)  # (N, D)

        # Compute cosine similarities between adjacent sentences
        similarities = self._adjacent_cosine_similarities(embeddings)

        # Find breakpoints: positions where similarity is low
        breakpoints = self._find_breakpoints(similarities)

        # Group sentences into chunks at breakpoints
        chunks = self._group_into_chunks(
            doc_id=doc_id,
            sentences=sentences,
            breakpoints=breakpoints,
            text=text,
            metadata=metadata,
        )

        logger.debug(
            "SemanticChunker: %s → %d chunks (breakpoints at %s)",
            doc_id,
            len(chunks),
            breakpoints,
        )
        return chunks

    @staticmethod
    def _adjacent_cosine_similarities(embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between each adjacent sentence pair."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        normalized = embeddings / norms
        # Dot product of adjacent rows
        return np.sum(normalized[:-1] * normalized[1:], axis=1)

    def _find_breakpoints(self, similarities: np.ndarray) -> List[int]:
        """
        Find sentence indices where chunks should break.

        A breakpoint at index i means sentences 0..i go in one chunk,
        sentences i+1..N go in the next.

        Uses the percentile of the similarity distribution: positions
        where similarity is in the lowest (100 - percentile)% are breakpoints.
        """
        threshold = float(np.percentile(similarities, 100 - self.breakpoint_percentile))
        breakpoints = [
            i for i, sim in enumerate(similarities) if sim < threshold
        ]
        return breakpoints

    def _group_into_chunks(
        self,
        doc_id: str,
        sentences: List[str],
        breakpoints: List[int],
        text: str,
        metadata: dict,
    ) -> List[Chunk]:
        """Group sentences into chunks, splitting further if over size limit."""
        # Create sentence groups based on breakpoints
        groups: List[List[str]] = []
        current_group: List[str] = []
        breakpoint_set = set(breakpoints)

        for i, sent in enumerate(sentences):
            current_group.append(sent)
            if i in breakpoint_set and current_group:
                groups.append(current_group)
                current_group = []
        if current_group:
            groups.append(current_group)

        # Convert groups to chunks, further splitting oversized groups
        chunks: List[Chunk] = []
        for group in groups:
            group_text = " ".join(group)
            if self._token_count(group_text) <= self.chunk_size:
                if self._token_count(group_text) >= self.min_chunk_size:
                    chunks.append(
                        self._make_chunk(
                            doc_id=doc_id,
                            text=group_text,
                            metadata=metadata,
                            chunk_index=len(chunks),
                        )
                    )
            else:
                # Group is too large; use sentence chunker to split it
                sub_doc = {"id": doc_id, "text": group_text, "metadata": metadata}
                sub_chunks = self._fallback.chunk(sub_doc)
                for sc in sub_chunks:
                    sc.chunk_id = f"{doc_id}_chunk_{len(chunks)}"
                    sc.metadata["chunk_index"] = len(chunks)
                    chunks.append(sc)

        return chunks


# ---------------------------------------------------------------------------
# Section-aware chunker
# ---------------------------------------------------------------------------

class SectionAwareChunker(BaseChunker):
    """
    Chunk clinical notes while preserving section boundaries.

    Clinical notes are naturally divided into sections (HPI, Assessment, Plan).
    This chunker respects those boundaries: chunks never span section boundaries,
    and each chunk is labeled with its source section.

    Intended for use with MIMIC-III ClinicalNote objects that have been
    pre-segmented into sections by ClinicalNoteSectionSegmenter.

    Args:
        inner_chunker: Chunker used within each section. Defaults to SentenceChunker.
        chunk_size: Token budget passed to inner_chunker.
        overlap: Overlap tokens passed to inner_chunker.

    Example::

        chunker = SectionAwareChunker()
        # document should have metadata["sections"] from ClinicalNote.to_rag_documents()
        chunks = chunker.chunk_from_note(note)
    """

    def __init__(
        self,
        inner_chunker: Optional[BaseChunker] = None,
        chunk_size: int = 384,
        overlap: int = 48,
    ):
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.inner_chunker = inner_chunker or SentenceChunker(
            chunk_size=chunk_size, overlap=overlap
        )

    def chunk(self, document: dict) -> List[Chunk]:
        """
        Chunk a single-section document.

        For multi-section notes, use chunk_from_note_sections() instead.
        Falls through to inner_chunker.
        """
        return self.inner_chunker.chunk(document)

    def chunk_from_note_sections(
        self, sections: List[dict], note_metadata: dict
    ) -> List[Chunk]:
        """
        Chunk each section independently, preserving section labels.

        Args:
            sections: List of section dicts with keys 'section_name', 'text'.
            note_metadata: Note-level metadata to attach to all chunks.

        Returns:
            Flat list of chunks, each labeled with its source section.
        """
        all_chunks: List[Chunk] = []
        for section in sections:
            section_name = section.get("section_name", "unknown")
            section_text = section.get("text", "")
            if not section_text.strip():
                continue

            section_doc = {
                "id": f"{note_metadata.get('note_id', 'note')}_{section_name}",
                "text": section_text,
                "metadata": {
                    **note_metadata,
                    "section": section_name,
                },
            }
            section_chunks = self.inner_chunker.chunk(section_doc)
            all_chunks.extend(section_chunks)

        # Re-index chunks globally
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = f"{note_metadata.get('note_id', 'note')}_chunk_{i}"
            chunk.metadata["chunk_index"] = i

        logger.debug(
            "SectionAwareChunker: note %s → %d chunks across %d sections",
            note_metadata.get("note_id"),
            len(all_chunks),
            len(sections),
        )
        return all_chunks


# ---------------------------------------------------------------------------
# Chunker factory
# ---------------------------------------------------------------------------

class ChunkerFactory:
    """Factory for instantiating chunkers from configuration."""

    @staticmethod
    def create(
        strategy: str,
        chunk_size: int = 512,
        overlap: int = 64,
        embed_fn=None,
        **kwargs,
    ) -> BaseChunker:
        """
        Create a chunker by strategy name.

        Args:
            strategy: One of 'sentence', 'semantic', 'section'.
            chunk_size: Token budget per chunk.
            overlap: Overlap tokens between chunks.
            embed_fn: Embedding function (required for 'semantic').

        Returns:
            BaseChunker instance.
        """
        strategy = strategy.lower()
        if strategy == "sentence":
            return SentenceChunker(chunk_size=chunk_size, overlap=overlap, **kwargs)
        elif strategy == "semantic":
            return SemanticChunker(
                embed_fn=embed_fn, chunk_size=chunk_size, **kwargs
            )
        elif strategy == "section":
            return SectionAwareChunker(chunk_size=chunk_size, overlap=overlap, **kwargs)
        else:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                "Choose from: 'sentence', 'semantic', 'section'."
            )
