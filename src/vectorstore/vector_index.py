"""
Vector store implementations for biomedical RAG.

Supports three backends:
  1. FAISSIndex — Facebook AI Similarity Search (flat/IVF/HNSW variants).
     Best for large-scale (>1M) in-memory retrieval with GPU acceleration.
  2. ChromaDBIndex — ChromaDB persistent vector database.
     Best for persistent multi-collection deployments with rich metadata filtering.
  3. HybridIndex — Combines FAISS dense retrieval with BM25 sparse retrieval,
     fused via Reciprocal Rank Fusion (RRF). Best overall retrieval performance.

Index versioning: each index is saved with a manifest file tracking:
  - Model name and embedding dim
  - Number of documents indexed
  - Creation timestamp
  - Chunking configuration

References:
  - Johnson, J. et al. (2019). Billion-scale similarity search with GPUs. IEEE T-BD.
    https://arxiv.org/abs/1702.08734
  - Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework:
    BM25 and Beyond. Foundations and Trends in Information Retrieval.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import log
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval result model
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single retrieved document with its score and metadata."""

    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0
    retrieval_method: str = "dense"  # 'dense', 'sparse', 'hybrid'

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Index manifest
# ---------------------------------------------------------------------------

@dataclass
class IndexManifest:
    """Metadata about a saved index."""

    model_name: str
    embedding_dim: int
    num_documents: int
    index_type: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "IndexManifest":
        with open(path) as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------------
# Abstract base index
# ---------------------------------------------------------------------------

class BaseVectorIndex(ABC):
    """Abstract base class for all vector index backends."""

    @abstractmethod
    def add_documents(
        self,
        documents: List[dict],
        embeddings: np.ndarray,
    ) -> None:
        """Add documents and their embeddings to the index."""

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> List[RetrievalResult]:
        """Search the index by query embedding."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Persist the index to disk."""

    @abstractmethod
    def load(self, directory: Path) -> None:
        """Load the index from disk."""

    @property
    @abstractmethod
    def num_documents(self) -> int:
        """Number of indexed documents."""


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

class FAISSIndex(BaseVectorIndex):
    """
    FAISS-based vector index for biomedical document retrieval.

    Index types:
      - flat: Exact nearest neighbor search. No approximation. Best for <100K docs.
      - ivf: Inverted file index. Approximate, fast for 100K–10M docs.
            Requires training (call train() before add_documents()).
      - hnsw: Hierarchical Navigable Small World. Approximate, very fast.
              Good for real-time serving.

    Args:
        index_type: One of 'flat', 'ivf', 'hnsw'.
        embedding_dim: Embedding dimensionality (768 for BERT models).
        nlist: Number of IVF cells (for index_type='ivf'). Typically sqrt(N).
        nprobe: IVF cells to probe at query time (higher = more accurate, slower).
        m: HNSW connectivity parameter (higher = more accurate, more memory).
        ef_construction: HNSW construction quality (higher = better graph).
        use_gpu: Use GPU FAISS if available (requires faiss-gpu).

    Example::

        index = FAISSIndex(index_type='ivf', embedding_dim=768, nlist=100)
        index.train(embeddings)
        index.add_documents(docs, embeddings)
        results = index.search(query_embedding, top_k=10)
        index.save(Path("data/indices/pubmed_ivf/"))
    """

    def __init__(
        self,
        index_type: str = "flat",
        embedding_dim: int = 768,
        nlist: int = 100,
        nprobe: int = 10,
        m: int = 16,
        ef_construction: int = 200,
        use_gpu: bool = False,
    ):
        self.index_type = index_type
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.ef_construction = ef_construction
        self.use_gpu = use_gpu

        self._index = None
        self._documents: List[dict] = []  # stores text + metadata
        self._is_trained = False

        self._build_index()

    def _build_index(self) -> None:
        """Build the FAISS index structure."""
        try:
            import faiss

            if self.index_type == "flat":
                self._index = faiss.IndexFlatIP(self.embedding_dim)  # inner product
                self._is_trained = True
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self._index = faiss.IndexIVFFlat(
                    quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT
                )
                self._index.nprobe = self.nprobe
            elif self.index_type == "hnsw":
                self._index = faiss.IndexHNSWFlat(
                    self.embedding_dim, self.m, faiss.METRIC_INNER_PRODUCT
                )
                self._index.hnsw.efConstruction = self.ef_construction
                self._is_trained = True
            else:
                raise ValueError(f"Unknown index_type: {self.index_type}")

            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                    logger.info("FAISS: using GPU")
                except Exception:
                    logger.warning("FAISS GPU not available, using CPU")

        except ImportError as exc:
            raise ImportError(
                "faiss-cpu or faiss-gpu required. "
                "Install: pip install faiss-cpu"
            ) from exc

    def train(self, embeddings: np.ndarray) -> None:
        """
        Train the IVF index (required before adding documents).

        Args:
            embeddings: Training vectors, shape (N, embedding_dim). N >= nlist.
        """
        if self.index_type != "ivf":
            self._is_trained = True
            return
        if embeddings.shape[0] < self.nlist:
            raise ValueError(
                f"Need at least {self.nlist} training vectors for IVF index, "
                f"got {embeddings.shape[0]}."
            )
        logger.info("Training IVF index on %d vectors...", len(embeddings))
        self._index.train(embeddings.astype(np.float32))
        self._is_trained = True
        logger.info("IVF index trained.")

    def add_documents(
        self,
        documents: List[dict],
        embeddings: np.ndarray,
    ) -> None:
        if not self._is_trained:
            logger.info("Auto-training IVF index on provided embeddings...")
            self.train(embeddings)

        self._documents.extend(documents)
        self._index.add(embeddings.astype(np.float32))
        logger.info(
            "Added %d documents. Total: %d", len(documents), len(self._documents)
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> List[RetrievalResult]:
        if len(self._documents) == 0:
            return []

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        # Fetch extra results if we'll be filtering
        fetch_k = top_k * 4 if filter_metadata else top_k
        fetch_k = min(fetch_k, len(self._documents))

        scores, indices = self._index.search(query_vec, fetch_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            doc = self._documents[idx]
            if filter_metadata and not self._matches_filter(doc.get("metadata", {}), filter_metadata):
                continue
            results.append(
                RetrievalResult(
                    chunk_id=doc.get("id", str(idx)),
                    text=doc.get("text", ""),
                    score=float(score),
                    metadata=doc.get("metadata", {}),
                    retrieval_method="dense",
                )
            )
            if len(results) >= top_k:
                break

        for i, r in enumerate(results):
            r.rank = i + 1
        return results

    @staticmethod
    def _matches_filter(metadata: dict, filter_criteria: dict) -> bool:
        """Check if document metadata matches all filter criteria."""
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            meta_val = metadata[key]
            if isinstance(value, list):
                if meta_val not in value:
                    return False
            elif meta_val != value:
                return False
        return True

    def save(self, directory: Path) -> None:
        import faiss
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        cpu_index = self._index
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self._index)
        faiss.write_index(cpu_index, str(directory / "faiss.index"))

        # Save document store
        with open(directory / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)

        # Save manifest
        manifest = IndexManifest(
            model_name="unknown",
            embedding_dim=self.embedding_dim,
            num_documents=len(self._documents),
            index_type=f"faiss_{self.index_type}",
            config={"nlist": self.nlist, "nprobe": self.nprobe},
        )
        manifest.save(directory / "manifest.json")
        logger.info("FAISS index saved to %s", directory)

    def load(self, directory: Path) -> None:
        import faiss
        directory = Path(directory)

        self._index = faiss.read_index(str(directory / "faiss.index"))
        if self.index_type == "ivf":
            self._index.nprobe = self.nprobe

        with open(directory / "documents.pkl", "rb") as f:
            self._documents = pickle.load(f)

        self._is_trained = True
        logger.info(
            "FAISS index loaded from %s (%d docs)", directory, len(self._documents)
        )

    @property
    def num_documents(self) -> int:
        return len(self._documents)


# ---------------------------------------------------------------------------
# BM25 sparse index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    BM25 (Best Match 25) sparse retrieval index.

    BM25 is a probabilistic bag-of-words ranking function that outperforms
    TF-IDF by normalizing for document length and using saturating term
    frequency.

    Key parameters (Robertson & Zaragoza 2009):
      - k1 (1.2–2.0): Controls term frequency saturation. Higher = more weight
                       on repeated terms.
      - b (0–1): Length normalization. 0 = no normalization, 1 = full normalization.
        For biomedical text, b=0.75 is standard.

    For biomedical text, we add clinical stopwords to avoid matching on
    ubiquitous clinical terms like "patient", "history", "diagnosis".

    Args:
        k1: BM25 k1 parameter.
        b: BM25 b parameter (length normalization).
        tokenizer: Tokenization function. Defaults to simple whitespace split.
    """

    CLINICAL_STOPWORDS = {
        "patient", "patients", "history", "diagnosis", "treatment", "clinical",
        "medical", "hospital", "care", "health", "disease", "condition",
        "symptoms", "reported", "noted", "presented", "was", "were", "the",
        "and", "of", "in", "to", "a", "an", "is", "are", "for", "with",
        "that", "this", "has", "have", "had", "been", "not", "no",
    }

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer=None,
    ):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or self._default_tokenize

        self._documents: List[dict] = []
        self._tokenized_docs: List[List[str]] = []
        self._df: Dict[str, int] = defaultdict(int)  # document frequency
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0
        self._is_built = False

    def _default_tokenize(self, text: str) -> List[str]:
        """Lowercase, remove punctuation, filter stopwords."""
        tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{1,30}\b", text.lower())
        return [t for t in tokens if t not in self.CLINICAL_STOPWORDS]

    def add_documents(self, documents: List[dict]) -> None:
        """Add documents to the BM25 index."""
        self._documents.extend(documents)
        for doc in documents:
            tokens = self.tokenizer(doc.get("text", ""))
            self._tokenized_docs.append(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._df[token] += 1
        self._build_idf()
        self._avgdl = np.mean([len(t) for t in self._tokenized_docs]) if self._tokenized_docs else 1.0
        self._is_built = True
        logger.debug("BM25: indexed %d documents", len(self._documents))

    def _build_idf(self) -> None:
        """Compute IDF for all terms."""
        N = len(self._documents)
        for term, df in self._df.items():
            self._idf[term] = log((N - df + 0.5) / (df + 0.5) + 1)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> List[RetrievalResult]:
        """
        BM25 search for query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter dict.

        Returns:
            List of RetrievalResult sorted by BM25 score (descending).
        """
        if not self._is_built:
            logger.warning("BM25 index is empty.")
            return []

        query_tokens = self.tokenizer(query)
        scores = np.zeros(len(self._documents))

        for token in query_tokens:
            if token not in self._idf:
                continue
            idf = self._idf[token]
            for i, doc_tokens in enumerate(self._tokenized_docs):
                tf = doc_tokens.count(token)
                dl = len(doc_tokens)
                score = idf * (
                    (tf * (self.k1 + 1))
                    / (tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
                )
                scores[i] += score

        # Sort by score
        sorted_indices = np.argsort(-scores)
        results = []
        for idx in sorted_indices:
            if scores[idx] <= 0:
                break
            doc = self._documents[idx]
            if filter_metadata and not FAISSIndex._matches_filter(
                doc.get("metadata", {}), filter_metadata
            ):
                continue
            results.append(
                RetrievalResult(
                    chunk_id=doc.get("id", str(idx)),
                    text=doc.get("text", ""),
                    score=float(scores[idx]),
                    metadata=doc.get("metadata", {}),
                    retrieval_method="sparse",
                )
            )
            if len(results) >= top_k:
                break

        for i, r in enumerate(results):
            r.rank = i + 1
        return results

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "bm25.pkl", "wb") as f:
            pickle.dump(
                {
                    "documents": self._documents,
                    "tokenized_docs": self._tokenized_docs,
                    "df": dict(self._df),
                    "idf": self._idf,
                    "avgdl": self._avgdl,
                    "k1": self.k1,
                    "b": self.b,
                },
                f,
            )

    def load(self, directory: Path) -> None:
        with open(Path(directory) / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self._documents = data["documents"]
        self._tokenized_docs = data["tokenized_docs"]
        self._df = defaultdict(int, data["df"])
        self._idf = data["idf"]
        self._avgdl = data["avgdl"]
        self.k1 = data["k1"]
        self.b = data["b"]
        self._is_built = True

    @property
    def num_documents(self) -> int:
        return len(self._documents)


# ---------------------------------------------------------------------------
# Hybrid index (Dense + Sparse with RRF)
# ---------------------------------------------------------------------------

class HybridIndex:
    """
    Hybrid vector index combining FAISS dense and BM25 sparse retrieval.

    Uses Reciprocal Rank Fusion (RRF) to merge dense and sparse rankings:
        RRF_score(d) = sum_r [ 1 / (k + rank_r(d)) ]

    where k=60 is a smoothing constant, and r iterates over retrieval methods.
    RRF is parameter-free and robust across different score scales.

    Research shows hybrid retrieval consistently outperforms either dense or
    sparse retrieval alone on biomedical benchmarks (Jin et al. 2023).

    Args:
        dense_index: FAISSIndex for dense retrieval.
        sparse_index: BM25Index for sparse retrieval.
        rrf_k: RRF smoothing constant (default 60 from Cormack et al. 2009).
        dense_weight: Weight for dense results in final fusion.
        sparse_weight: Weight for sparse results in final fusion.

    Example::

        index = HybridIndex()
        index.add_documents(docs, embeddings)
        results = index.search(query_text, query_embedding, top_k=10)
    """

    def __init__(
        self,
        dense_index: Optional[FAISSIndex] = None,
        sparse_index: Optional[BM25Index] = None,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        embedding_dim: int = 768,
    ):
        self.dense_index = dense_index or FAISSIndex(
            index_type="flat", embedding_dim=embedding_dim
        )
        self.sparse_index = sparse_index or BM25Index()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def add_documents(
        self,
        documents: List[dict],
        embeddings: np.ndarray,
    ) -> None:
        """Add documents to both dense and sparse indices."""
        self.dense_index.add_documents(documents, embeddings)
        self.sparse_index.add_documents(documents)
        logger.info("HybridIndex: %d documents indexed", len(documents))

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
        fetch_multiplier: int = 3,
    ) -> List[RetrievalResult]:
        """
        Hybrid search using RRF fusion of dense and sparse results.

        Args:
            query_text: Raw query text (for BM25).
            query_embedding: Query embedding vector (for FAISS).
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter.
            fetch_multiplier: Fetch top_k * multiplier from each sub-index
                              before fusion to ensure coverage.

        Returns:
            List of RetrievalResult sorted by RRF score (descending).
        """
        fetch_k = top_k * fetch_multiplier

        dense_results = self.dense_index.search(
            query_embedding, top_k=fetch_k, filter_metadata=filter_metadata
        )
        sparse_results = self.sparse_index.search(
            query_text, top_k=fetch_k, filter_metadata=filter_metadata
        )

        return self._reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k,
        )

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        Args:
            dense_results: Dense retrieval results (rank-ordered).
            sparse_results: Sparse retrieval results (rank-ordered).
            top_k: Number of results to return.

        Returns:
            Merged and re-ranked results.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, RetrievalResult] = {}

        # Dense results
        for rank, result in enumerate(dense_results, 1):
            rrf_scores[result.chunk_id] += (
                self.dense_weight / (self.rrf_k + rank)
            )
            doc_map[result.chunk_id] = result

        # Sparse results
        for rank, result in enumerate(sparse_results, 1):
            rrf_scores[result.chunk_id] += (
                self.sparse_weight / (self.rrf_k + rank)
            )
            if result.chunk_id not in doc_map:
                doc_map[result.chunk_id] = result

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])
        merged = []
        for i, chunk_id in enumerate(sorted_ids[:top_k]):
            result = doc_map[chunk_id]
            merged.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=result.text,
                    score=rrf_scores[chunk_id],
                    metadata=result.metadata,
                    rank=i + 1,
                    retrieval_method="hybrid_rrf",
                )
            )
        return merged

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.dense_index.save(directory / "dense")
        self.sparse_index.save(directory / "sparse")
        with open(directory / "hybrid_config.json", "w") as f:
            json.dump(
                {
                    "rrf_k": self.rrf_k,
                    "dense_weight": self.dense_weight,
                    "sparse_weight": self.sparse_weight,
                },
                f,
            )
        logger.info("HybridIndex saved to %s", directory)

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        self.dense_index.load(directory / "dense")
        self.sparse_index.load(directory / "sparse")
        with open(directory / "hybrid_config.json") as f:
            config = json.load(f)
        self.rrf_k = config["rrf_k"]
        self.dense_weight = config["dense_weight"]
        self.sparse_weight = config["sparse_weight"]
        logger.info("HybridIndex loaded from %s", directory)

    @property
    def num_documents(self) -> int:
        return self.dense_index.num_documents
