"""
Retrieval pipeline for biomedical RAG.

Implements three retrieval modes:
  1. DenseRetriever — pure dense retrieval (FAISS) with optional cross-encoder
     re-ranking. Fast; struggles with exact medical terminology.
  2. SparseRetriever — pure BM25. Excellent for exact medical term matching
     (drug names, gene symbols, UMLS codes) but misses semantic similarity.
  3. HybridRetriever — combines dense + sparse via RRF, then re-ranks top
     candidates with a cross-encoder. Best overall performance on BioASQ/MedQA.

Query expansion:
  Uses UMLS (Unified Medical Language System) synonyms to expand biomedical
  queries. "heart attack" → ["myocardial infarction", "MI", "acute MI", ...].
  Requires UMLS Metathesaurus installation (umls.nlm.nih.gov/download).

Cross-encoder re-ranking:
  After initial retrieval, a cross-encoder (ms-marco-MiniLM-L-12 or MedCPT
  cross-encoder) re-scores candidate documents by jointly encoding query and
  document. Significantly improves precision at the cost of ~3ms/candidate.

Retrieval evaluation:
  compute_retrieval_metrics() calculates precision@k, recall@k, MRR, and
  NDCG@k for a set of queries with ground truth relevant document IDs.

References:
  - Lin, J. et al. (2021). Pretrained Transformers for Text Ranking:
    BERT and Beyond. Morgan & Claypool.
  - Cormack, G. et al. (2009). Reciprocal Rank Fusion Outperforms Condorcet.
    SIGIR'09.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.embeddings.biomedical_embedder import BiomedicalEmbedder
from src.vectorstore.vector_index import (
    BM25Index,
    FAISSIndex,
    HybridIndex,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UMLS query expansion
# ---------------------------------------------------------------------------

class UMLSQueryExpander:
    """
    Expand biomedical queries using UMLS synonyms.

    UMLS (Unified Medical Language System) maps biomedical concepts to
    Concept Unique Identifiers (CUIs) with thousands of synonymous terms.
    Expanding "heart attack" to include "myocardial infarction", "MI",
    "acute MI" dramatically improves BM25 recall for medical queries.

    Requires QuickUMLS or a UMLS API key. Falls back to a curated synonym
    dictionary if full UMLS is unavailable.

    Args:
        use_quickumls: If True, use QuickUMLS for expansion (requires installation).
        use_fallback: If True, use built-in curated synonym dict.
        max_synonyms: Maximum synonyms to add per term.
        threshold: QuickUMLS similarity threshold.
    """

    # Curated synonym fallback for common clinical terms
    _FALLBACK_SYNONYMS: Dict[str, List[str]] = {
        "heart attack": ["myocardial infarction", "MI", "acute MI", "STEMI", "NSTEMI"],
        "myocardial infarction": ["heart attack", "MI", "acute MI", "cardiac infarction"],
        "stroke": ["cerebrovascular accident", "CVA", "cerebral infarction", "brain attack"],
        "cancer": ["malignancy", "neoplasm", "tumor", "carcinoma"],
        "diabetes": ["diabetes mellitus", "DM", "type 2 diabetes", "T2DM", "hyperglycemia"],
        "hypertension": ["high blood pressure", "HTN", "elevated blood pressure"],
        "blood pressure": ["BP", "hypertension", "hypotension"],
        "drug interaction": ["DDI", "drug-drug interaction", "pharmacokinetic interaction"],
        "adverse event": ["adverse drug reaction", "ADR", "side effect", "drug toxicity"],
        "chest pain": ["angina", "angina pectoris", "thoracic pain", "precordial pain"],
        "shortness of breath": ["dyspnea", "breathlessness", "respiratory distress"],
        "kidney disease": ["renal disease", "nephropathy", "chronic kidney disease", "CKD"],
        "alzheimer": ["alzheimer's disease", "AD", "dementia", "cognitive decline"],
        "covid": ["COVID-19", "SARS-CoV-2", "coronavirus disease", "coronavirus"],
        "mri": ["magnetic resonance imaging", "MRI scan", "NMR imaging"],
        "ct scan": ["computed tomography", "CT", "CAT scan", "computerized tomography"],
    }

    def __init__(
        self,
        use_quickumls: bool = False,
        use_fallback: bool = True,
        max_synonyms: int = 5,
        threshold: float = 0.9,
    ):
        self.use_quickumls = use_quickumls
        self.use_fallback = use_fallback
        self.max_synonyms = max_synonyms
        self.threshold = threshold
        self._matcher = None

        if use_quickumls:
            self._load_quickumls()

    def _load_quickumls(self) -> None:
        """Load QuickUMLS matcher."""
        try:
            from quickumls import QuickUMLS
            quickumls_path = os.getenv("QUICKUMLS_PATH", "/usr/local/share/quickumls")
            self._matcher = QuickUMLS(quickumls_path, threshold=self.threshold)
            logger.info("QuickUMLS loaded from %s", quickumls_path)
        except ImportError:
            logger.warning("QuickUMLS not installed. Falling back to curated synonyms.")
            self.use_quickumls = False

    def expand(self, query: str) -> List[str]:
        """
        Return expanded query variants including UMLS synonyms.

        Args:
            query: Original query string.

        Returns:
            List of query strings (original + expanded variants).
        """
        queries = [query]
        additional_terms: List[str] = []

        if self.use_quickumls and self._matcher:
            try:
                matches = self._matcher.match(query)
                for match in matches[:3]:  # Top 3 concept matches
                    for ngram_match in match:
                        synonyms = ngram_match.get("similarity_score_list", [])
                        for syn in synonyms[: self.max_synonyms]:
                            term = syn.get("term", "")
                            if term and term.lower() not in query.lower():
                                additional_terms.append(term)
            except Exception as exc:
                logger.warning("QuickUMLS expansion failed: %s", exc)

        if self.use_fallback:
            query_lower = query.lower()
            for key, synonyms in self._FALLBACK_SYNONYMS.items():
                if key in query_lower:
                    for syn in synonyms[: self.max_synonyms]:
                        if syn.lower() not in query_lower:
                            additional_terms.append(syn)

        # Add an expanded query string with synonyms appended
        if additional_terms:
            expanded = query + " " + " ".join(additional_terms[:self.max_synonyms])
            queries.append(expanded)

        return queries


# ---------------------------------------------------------------------------
# Cross-encoder re-ranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Re-rank candidate documents using a cross-encoder model.

    Cross-encoders jointly encode the query and document together, giving
    much more accurate relevance scores than bi-encoder (embedding) models
    at the cost of ~O(N) inference calls.

    Recommended usage: retrieve top-50 with bi-encoder, re-rank to get top-10.

    Supported models:
      - cross-encoder/ms-marco-MiniLM-L-12-v2: General, fast (12 layers)
      - cross-encoder/ms-marco-electra-base: More accurate, slower
      - ncbi/MedCPT-Cross-Encoder: Biomedical-specific (best for clinical use)

    Args:
        model_name: Cross-encoder model name.
        device: Inference device.
        batch_size: Re-ranking batch size.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

        if device == "auto":
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = device

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self._device,
            )
            logger.info("CrossEncoder loaded: %s", self.model_name)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers required for re-ranking. "
                "Install with: pip install sentence-transformers"
            ) from exc

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Re-rank candidate documents by cross-encoder score.

        Args:
            query: Query string.
            candidates: Initial retrieval results.
            top_k: Return only top_k results. None = return all.

        Returns:
            Re-ranked list of RetrievalResult with updated scores and ranks.
        """
        if not candidates:
            return []

        self._load_model()
        pairs = [(query, c.text) for c in candidates]

        # Batch inference
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            batch_scores = self._model.predict(batch_pairs)
            scores.extend(batch_scores)

        # Sort by cross-encoder score
        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_k = top_k or len(scored)
        reranked = []
        for rank, (result, score) in enumerate(scored[:top_k], 1):
            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=float(score),
                    metadata=result.metadata,
                    rank=rank,
                    retrieval_method=f"{result.retrieval_method}+reranked",
                )
            )
        return reranked


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""

    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    num_queries: int = 0
    avg_latency_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Retrieval Metrics ({self.num_queries} queries):",
            f"  MRR:          {self.mrr:.4f}",
        ]
        for k in sorted(self.precision_at_k.keys()):
            lines.append(
                f"  P@{k}: {self.precision_at_k[k]:.4f}  "
                f"R@{k}: {self.recall_at_k.get(k, 0):.4f}  "
                f"NDCG@{k}: {self.ndcg_at_k.get(k, 0):.4f}"
            )
        lines.append(f"  Avg latency:  {self.avg_latency_ms:.1f} ms/query")
        return "\n".join(lines)


def compute_retrieval_metrics(
    retrieved_ids: List[List[str]],
    relevant_ids: List[List[str]],
    k_values: List[int] = [1, 5, 10],
    latencies_ms: Optional[List[float]] = None,
) -> RetrievalMetrics:
    """
    Compute retrieval evaluation metrics.

    Args:
        retrieved_ids: For each query, list of retrieved document IDs (ranked).
        relevant_ids: For each query, list of ground-truth relevant IDs.
        k_values: Values of k to compute metrics at.
        latencies_ms: Per-query latencies in milliseconds.

    Returns:
        RetrievalMetrics object with precision@k, recall@k, MRR, NDCG@k.
    """
    n = len(retrieved_ids)
    assert len(relevant_ids) == n, "Mismatch: retrieved and relevant lists must match."

    precision_at_k = {k: 0.0 for k in k_values}
    recall_at_k = {k: 0.0 for k in k_values}
    ndcg_at_k = {k: 0.0 for k in k_values}
    mrr_sum = 0.0

    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        relevant_set = set(relevant)
        if not relevant_set:
            continue

        # MRR: rank of first relevant result
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                mrr_sum += 1.0 / rank
                break

        for k in k_values:
            retrieved_k = retrieved[:k]
            hits = sum(1 for d in retrieved_k if d in relevant_set)

            precision_at_k[k] += hits / k
            recall_at_k[k] += hits / len(relevant_set)

            # NDCG@k
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, doc_id in enumerate(retrieved_k, 1)
                if doc_id in relevant_set
            )
            ideal_hits = min(k, len(relevant_set))
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg_at_k[k] += dcg / idcg if idcg > 0 else 0.0

    metrics = RetrievalMetrics(
        precision_at_k={k: v / n for k, v in precision_at_k.items()},
        recall_at_k={k: v / n for k, v in recall_at_k.items()},
        mrr=mrr_sum / n,
        ndcg_at_k={k: v / n for k, v in ndcg_at_k.items()},
        num_queries=n,
        avg_latency_ms=sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
    )
    return metrics


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Full retrieval pipeline: hybrid (dense+BM25) + cross-encoder re-ranking.

    This is the primary retriever for production clinical use. Steps:
      1. Optionally expand query with UMLS synonyms.
      2. Retrieve top-N candidates via HybridIndex (RRF of dense + BM25).
      3. Re-rank top candidates with cross-encoder.
      4. Return top-K final results.

    Args:
        index: HybridIndex instance (pre-built or loaded from disk).
        embedder: BiomedicalEmbedder for query encoding.
        reranker: CrossEncoderReranker (optional; None = skip re-ranking).
        query_expander: UMLSQueryExpander (optional).
        initial_fetch_k: Candidates to fetch from index before re-ranking.
        config: Optional config dict to override defaults.

    Example::

        retriever = HybridRetriever.from_config("configs/pubmed_rag_config.yaml")
        results = retriever.retrieve(
            "What is the mechanism of metformin in type 2 diabetes?",
            top_k=5,
        )
        for r in results:
            print(f"[{r.rank}] {r.score:.3f} — {r.text[:100]}")
    """

    def __init__(
        self,
        index: HybridIndex,
        embedder: BiomedicalEmbedder,
        reranker: Optional[CrossEncoderReranker] = None,
        query_expander: Optional[UMLSQueryExpander] = None,
        initial_fetch_k: int = 50,
    ):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.query_expander = query_expander
        self.initial_fetch_k = initial_fetch_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
        expand_query: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-K relevant documents for a query.

        Args:
            query: Natural language query.
            top_k: Number of final results to return.
            filter_metadata: Optional metadata filter (e.g., {'source': 'pubmed'}).
            expand_query: If True and query_expander is set, expand query.

        Returns:
            List of RetrievalResult, ranked by relevance.
        """
        start = time.time()

        # Query expansion
        primary_query = query
        if expand_query and self.query_expander:
            expanded = self.query_expander.expand(query)
            primary_query = expanded[0] if expanded else query
            if len(expanded) > 1:
                # Use the richer expanded query for BM25
                primary_query = expanded[1]

        # Encode query
        query_embedding = self.embedder.encode_queries([query])[0]

        # Hybrid retrieval
        candidates = self.index.search(
            query_text=primary_query,
            query_embedding=query_embedding,
            top_k=self.initial_fetch_k,
            filter_metadata=filter_metadata,
        )

        # Re-ranking
        if self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        latency = (time.time() - start) * 1000
        logger.debug(
            "retrieve('%s'): %d results in %.1fms", query[:60], len(candidates), latency
        )
        return candidates

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve for multiple queries.

        Args:
            queries: List of query strings.
            top_k: Results per query.
            filter_metadata: Metadata filter applied to all queries.

        Returns:
            List of result lists, one per query.
        """
        return [
            self.retrieve(q, top_k=top_k, filter_metadata=filter_metadata)
            for q in queries
        ]

    def evaluate(
        self,
        queries: List[str],
        relevant_doc_ids: List[List[str]],
        top_k: int = 10,
        k_values: List[int] = [1, 5, 10],
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval on a set of queries with known relevant documents.

        Args:
            queries: Evaluation queries.
            relevant_doc_ids: For each query, list of relevant document IDs.
            top_k: Retrieve this many candidates per query.
            k_values: k values for precision/recall/NDCG computation.

        Returns:
            RetrievalMetrics with precision@k, recall@k, MRR, NDCG@k.
        """
        all_retrieved_ids: List[List[str]] = []
        latencies: List[float] = []

        for query in queries:
            start = time.time()
            results = self.retrieve(query, top_k=max(k_values))
            latencies.append((time.time() - start) * 1000)
            all_retrieved_ids.append([r.chunk_id for r in results])

        return compute_retrieval_metrics(
            retrieved_ids=all_retrieved_ids,
            relevant_ids=relevant_doc_ids,
            k_values=k_values,
            latencies_ms=latencies,
        )

    @classmethod
    def from_config(cls, config_path: str) -> "HybridRetriever":
        """
        Instantiate retriever from a YAML config file.

        Args:
            config_path: Path to pubmed_rag_config.yaml.

        Returns:
            HybridRetriever instance with index loaded from configured path.
        """
        import yaml
        from pathlib import Path

        with open(config_path) as f:
            config = yaml.safe_load(f)

        retrieval_cfg = config.get("retrieval", {})
        index_dir = Path(config.get("index", {}).get("path", "data/indices/default"))

        embedder = BiomedicalEmbedder(
            model_id=config.get("embedding", {}).get("model_id", "pubmedbert"),
            batch_size=config.get("embedding", {}).get("batch_size", 32),
        )

        index = HybridIndex(embedding_dim=embedder.dim)
        index.load(index_dir)

        reranker = None
        if retrieval_cfg.get("use_reranker", True):
            reranker = CrossEncoderReranker(
                model_name=retrieval_cfg.get(
                    "reranker_model", "cross-encoder/ms-marco-MiniLM-L-12-v2"
                )
            )

        query_expander = None
        if retrieval_cfg.get("use_query_expansion", True):
            query_expander = UMLSQueryExpander(
                use_fallback=True,
                use_quickumls=retrieval_cfg.get("use_quickumls", False),
            )

        return cls(
            index=index,
            embedder=embedder,
            reranker=reranker,
            query_expander=query_expander,
            initial_fetch_k=retrieval_cfg.get("initial_fetch_k", 50),
        )
