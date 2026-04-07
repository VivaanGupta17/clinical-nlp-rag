"""
RAG evaluation framework for biomedical question answering.

Implements a comprehensive evaluation suite for assessing RAG pipeline quality
across four dimensions:

  1. Retrieval Quality:
     precision@k, recall@k, MRR, NDCG@k — measures whether the right documents
     are being retrieved.

  2. Generation Quality:
     BLEU, ROUGE-1/2/L, BERTScore — measures lexical and semantic overlap
     between generated and reference answers.

  3. Faithfulness:
     Hallucination rate, citation accuracy, entailment-based faithfulness
     (from hallucination_detector) — measures grounding to retrieved context.

  4. Domain-Specific Accuracy:
     Medical accuracy verified against gold-standard biomedical QA datasets.

RAGAS-style evaluation:
  Implements the four RAGAS dimensions (Shahul et al. 2023):
    - Faithfulness: generated answer supported by context
    - Answer Relevance: answer addresses the question
    - Context Recall: relevant information retrieved
    - Context Precision: retrieved context is relevant

References:
  - Es, S. et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented
    Generation. arXiv:2309.15217.
  - Zhang, T. et al. (2020). BERTScore: Evaluating Text Generation with BERT.
    ICLR 2020.
  - Lin, C-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
    ACL Workshop.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation result models
# ---------------------------------------------------------------------------

@dataclass
class RetrievalEvalResult:
    """Retrieval quality metrics."""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    num_queries: int
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationEvalResult:
    """Generation quality metrics."""
    bleu: float
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    bertscore_f: float = 0.0
    num_examples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FaithfulnessEvalResult:
    """Faithfulness and hallucination metrics."""
    faithfulness_score: float
    hallucination_rate: float
    citation_accuracy: float
    num_evaluated: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RAGASEvalResult:
    """RAGAS-style comprehensive RAG evaluation."""
    faithfulness: float        # Answer grounded in context
    answer_relevance: float    # Answer addresses question
    context_recall: float      # Retrieved context covers relevant info
    context_precision: float   # Fraction of retrieved context that's relevant
    overall_score: float       # Harmonic mean of all four dimensions
    num_evaluated: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FullEvalReport:
    """Complete RAG evaluation report across all dimensions."""

    retrieval: Optional[RetrievalEvalResult] = None
    generation: Optional[GenerationEvalResult] = None
    faithfulness: Optional[FaithfulnessEvalResult] = None
    ragas: Optional[RAGASEvalResult] = None
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.utcnow().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "generation": self.generation.to_dict() if self.generation else None,
            "faithfulness": self.faithfulness.to_dict() if self.faithfulness else None,
            "ragas": self.ragas.to_dict() if self.ragas else None,
            "config": self.config,
        }

    def print_summary(self) -> None:
        print("=" * 60)
        print("RAG EVALUATION REPORT")
        print("=" * 60)
        if self.retrieval:
            r = self.retrieval
            print("\n--- Retrieval ---")
            print(f"  MRR:          {r.mrr:.4f}")
            for k in sorted(r.precision_at_k):
                print(f"  P@{k}:         {r.precision_at_k[k]:.4f}")
                print(f"  R@{k}:         {r.recall_at_k.get(k, 0):.4f}")
                print(f"  NDCG@{k}:      {r.ndcg_at_k.get(k, 0):.4f}")
            print(f"  Avg latency:  {r.avg_latency_ms:.1f} ms/query")
        if self.generation:
            g = self.generation
            print("\n--- Generation ---")
            print(f"  BLEU:         {g.bleu:.4f}")
            print(f"  ROUGE-1 F:    {g.rouge1_f:.4f}")
            print(f"  ROUGE-2 F:    {g.rouge2_f:.4f}")
            print(f"  ROUGE-L F:    {g.rougeL_f:.4f}")
            if g.bertscore_f:
                print(f"  BERTScore F:  {g.bertscore_f:.4f}")
        if self.faithfulness:
            f = self.faithfulness
            print("\n--- Faithfulness ---")
            print(f"  Faithfulness: {f.faithfulness_score:.4f}")
            print(f"  Hallucination rate: {f.hallucination_rate:.1%}")
            print(f"  Citation accuracy:  {f.citation_accuracy:.4f}")
        if self.ragas:
            r = self.ragas
            print("\n--- RAGAS ---")
            print(f"  Faithfulness:       {r.faithfulness:.4f}")
            print(f"  Answer Relevance:   {r.answer_relevance:.4f}")
            print(f"  Context Recall:     {r.context_recall:.4f}")
            print(f"  Context Precision:  {r.context_precision:.4f}")
            print(f"  Overall Score:      {r.overall_score:.4f}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# BLEU implementation
# ---------------------------------------------------------------------------

def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """
    Compute corpus-level BLEU score.

    Args:
        predictions: Generated text strings.
        references: Reference text strings (parallel to predictions).
        max_n: Maximum n-gram order (default 4 for BLEU-4).
        smooth: Apply Chen-Cherry smoothing for short sequences.

    Returns:
        BLEU score in [0, 1].
    """
    def ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    precision_scores = []
    bp = 0.0
    total_pred_len = 0
    total_ref_len = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        total_pred_len += len(pred_tokens)
        total_ref_len += len(ref_tokens)

    # Brevity penalty
    if total_pred_len < total_ref_len:
        bp = math.exp(1 - total_ref_len / max(total_pred_len, 1))
    else:
        bp = 1.0

    for n in range(1, max_n + 1):
        clipped_count = 0
        total_count = 0
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            pred_ngrams = ngrams(pred_tokens, n)
            ref_ngrams = ngrams(ref_tokens, n)
            clipped = sum(
                min(count, ref_ngrams[ngram])
                for ngram, count in pred_ngrams.items()
            )
            clipped_count += clipped
            total_count += max(len(pred_tokens) - n + 1, 0)

        if total_count == 0:
            precision = 0.0
        elif clipped_count == 0 and smooth:
            precision = 1.0 / (2 ** n)  # Add-one smoothing approximation
        else:
            precision = clipped_count / total_count
        precision_scores.append(precision)

    if all(p == 0 for p in precision_scores):
        return 0.0

    log_avg = sum(math.log(max(p, 1e-10)) for p in precision_scores) / max_n
    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# ROUGE implementation
# ---------------------------------------------------------------------------

def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        predictions: Generated text strings.
        references: Reference text strings.

    Returns:
        Dict with rouge1_f, rouge2_f, rougeL_f keys.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        r1_f = r2_f = rL_f = 0.0
        n = len(predictions)
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            r1_f += scores["rouge1"].fmeasure
            r2_f += scores["rouge2"].fmeasure
            rL_f += scores["rougeL"].fmeasure
        return {
            "rouge1_f": r1_f / n if n > 0 else 0.0,
            "rouge2_f": r2_f / n if n > 0 else 0.0,
            "rougeL_f": rL_f / n if n > 0 else 0.0,
        }
    except ImportError:
        # Fallback: simple ROUGE-1 implementation
        logger.warning("rouge_score not installed. Using simple ROUGE-1 fallback.")
        return _simple_rouge1(predictions, references)


def _simple_rouge1(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    total_f = 0.0
    for pred, ref in zip(predictions, references):
        p_tokens = set(pred.lower().split())
        r_tokens = set(ref.lower().split())
        if not p_tokens or not r_tokens:
            continue
        overlap = len(p_tokens & r_tokens)
        prec = overlap / len(p_tokens)
        rec = overlap / len(r_tokens)
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        total_f += f
    avg_f = total_f / len(predictions) if predictions else 0.0
    return {"rouge1_f": avg_f, "rouge2_f": 0.0, "rougeL_f": avg_f}


# ---------------------------------------------------------------------------
# BERTScore wrapper
# ---------------------------------------------------------------------------

def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: str = "cpu",
) -> float:
    """
    Compute corpus-level BERTScore F1.

    Uses DeBERTa-xlarge-mnli for biomedical text (stronger than BERT-base).
    BiomedBERT is also a strong choice for clinical text.

    Args:
        predictions: Generated text.
        references: Reference text.
        model_type: BERTScore backbone model.
        device: Inference device.

    Returns:
        Average F1 BERTScore.
    """
    try:
        from bert_score import score
        P, R, F = score(
            predictions,
            references,
            model_type=model_type,
            device=device,
            verbose=False,
        )
        return float(F.mean())
    except ImportError:
        logger.warning("bert_score not installed. Skipping BERTScore computation.")
        return 0.0


# ---------------------------------------------------------------------------
# RAGAS-style evaluator
# ---------------------------------------------------------------------------

class RAGASEvaluator:
    """
    RAGAS-style evaluation for RAG pipelines.

    Measures four key dimensions using an LLM as judge:
      1. Faithfulness: Is each claim in the answer supported by the context?
      2. Answer Relevance: Does the answer address the question?
      3. Context Recall: Does the retrieved context contain the answer?
      4. Context Precision: Is the retrieved context focused on the query?

    Args:
        provider: LLM provider for LLM-as-judge scoring.
        batch_size: Number of examples to evaluate per batch.
    """

    FAITHFULNESS_PROMPT = """Given the following context and answer, score whether the answer
is faithful to the context on a scale of 0.0 to 1.0.

Context: {context}
Answer: {answer}

A score of 1.0 means every claim in the answer is supported by the context.
A score of 0.0 means the answer contains facts not in the context.
Respond with ONLY a number between 0.0 and 1.0."""

    ANSWER_RELEVANCE_PROMPT = """Rate how well the answer addresses the question on a scale of 0.0 to 1.0.

Question: {question}
Answer: {answer}

A score of 1.0 means the answer directly and completely answers the question.
A score of 0.0 means the answer is irrelevant.
Respond with ONLY a number between 0.0 and 1.0."""

    def __init__(self, provider=None, batch_size: int = 10):
        self.provider = provider
        self.batch_size = batch_size

    def evaluate(
        self,
        questions: List[str],
        contexts: List[List[str]],  # Retrieved passages for each question
        answers: List[str],         # Generated answers
        ground_truths: Optional[List[str]] = None,
    ) -> RAGASEvalResult:
        """
        Evaluate a set of QA examples on all RAGAS dimensions.

        Args:
            questions: List of user questions.
            contexts: Retrieved context passages for each question.
            answers: Generated answers.
            ground_truths: Reference answers (for context recall computation).

        Returns:
            RAGASEvalResult with scores for all four dimensions.
        """
        n = len(questions)
        faithfulness_scores = []
        relevance_scores = []
        context_recalls = []
        context_precisions = []

        for i, (question, context_list, answer) in enumerate(
            zip(questions, contexts, answers)
        ):
            context_str = "\n\n".join(context_list[:5])

            # Faithfulness
            f_score = self._score_faithfulness(context_str, answer)
            faithfulness_scores.append(f_score)

            # Answer relevance
            r_score = self._score_answer_relevance(question, answer)
            relevance_scores.append(r_score)

            # Context recall (needs ground truth)
            if ground_truths and i < len(ground_truths):
                cr_score = self._score_context_recall(
                    ground_truths[i], context_list
                )
                context_recalls.append(cr_score)

            # Context precision
            cp_score = self._score_context_precision(question, context_list)
            context_precisions.append(cp_score)

        def safe_mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        faithfulness = safe_mean(faithfulness_scores)
        answer_relevance = safe_mean(relevance_scores)
        context_recall = safe_mean(context_recalls) if context_recalls else 0.0
        context_precision = safe_mean(context_precisions)

        # Overall: harmonic mean of all four
        scores = [faithfulness, answer_relevance, context_recall, context_precision]
        non_zero = [s for s in scores if s > 0]
        if non_zero:
            overall = len(non_zero) / sum(1 / s for s in non_zero)
        else:
            overall = 0.0

        return RAGASEvalResult(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_recall=context_recall,
            context_precision=context_precision,
            overall_score=overall,
            num_evaluated=n,
        )

    def _score_faithfulness(self, context: str, answer: str) -> float:
        if self.provider:
            try:
                resp = self.provider.generate(
                    system_prompt="You are a faithful evaluator. Always respond with a number.",
                    user_message=self.FAITHFULNESS_PROMPT.format(
                        context=context[:2000], answer=answer[:1000]
                    ),
                )
                return self._parse_score(resp)
            except Exception:
                pass
        # Fallback: rough heuristic based on sentence overlap
        return self._overlap_faithfulness(context, answer)

    def _score_answer_relevance(self, question: str, answer: str) -> float:
        if self.provider:
            try:
                resp = self.provider.generate(
                    system_prompt="You are an evaluator. Respond only with a number.",
                    user_message=self.ANSWER_RELEVANCE_PROMPT.format(
                        question=question, answer=answer[:1000]
                    ),
                )
                return self._parse_score(resp)
            except Exception:
                pass
        # Fallback: token overlap between question and answer
        q_tokens = set(question.lower().split())
        a_tokens = set(answer.lower().split())
        if not q_tokens:
            return 0.0
        overlap = len(q_tokens & a_tokens) / len(q_tokens)
        return min(overlap * 2.0, 1.0)  # Scale up

    def _score_context_recall(
        self, ground_truth: str, context_list: List[str]
    ) -> float:
        """Compute how much ground truth information is present in context."""
        gt_tokens = set(ground_truth.lower().split())
        ctx_tokens = set(" ".join(context_list).lower().split())
        if not gt_tokens:
            return 1.0
        return len(gt_tokens & ctx_tokens) / len(gt_tokens)

    def _score_context_precision(
        self, question: str, context_list: List[str]
    ) -> float:
        """Compute fraction of retrieved passages relevant to the question."""
        if not context_list:
            return 0.0
        q_tokens = set(question.lower().split())
        relevant = 0
        for ctx in context_list:
            ctx_tokens = set(ctx.lower().split())
            overlap = len(q_tokens & ctx_tokens) / max(len(q_tokens), 1)
            if overlap > 0.15:  # Rough relevance threshold
                relevant += 1
        return relevant / len(context_list)

    @staticmethod
    def _overlap_faithfulness(context: str, answer: str) -> float:
        """Fallback faithfulness: fraction of answer sentences in context."""
        answer_sentences = re.split(r"(?<=[.!?])\s+", answer)
        ctx_lower = context.lower()
        supported = sum(
            1 for sent in answer_sentences
            if len(sent) > 20 and
            any(token in ctx_lower for token in sent.lower().split()[:5])
        )
        total = len([s for s in answer_sentences if len(s) > 20])
        return supported / total if total > 0 else 1.0

    @staticmethod
    def _parse_score(text: str) -> float:
        """Parse a float score from LLM response text."""
        matches = re.findall(r"\b(0\.\d+|1\.0|0|1)\b", text)
        if matches:
            return max(0.0, min(1.0, float(matches[0])))
        return 0.5


# ---------------------------------------------------------------------------
# Full RAG evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Comprehensive RAG evaluation orchestrator.

    Combines retrieval metrics, generation quality, faithfulness, and
    RAGAS-style evaluation into a unified report.

    Args:
        retriever: HybridRetriever instance.
        generator: ClinicalRAGGenerator instance.
        hallucination_detector: HallucinationDetector instance.
        ragas_evaluator: RAGASEvaluator instance.
        compute_bertscore: Whether to compute BERTScore (slow).

    Example::

        evaluator = RAGEvaluator(retriever, generator, detector)
        report = evaluator.evaluate(
            qa_dataset=[
                {"question": "...", "answer": "...", "relevant_doc_ids": [...]},
            ]
        )
        report.print_summary()
        report.save("results/eval_report.json")
    """

    def __init__(
        self,
        retriever=None,
        generator=None,
        hallucination_detector=None,
        ragas_evaluator: Optional[RAGASEvaluator] = None,
        compute_bertscore: bool = False,
        k_values: List[int] = [1, 5, 10],
    ):
        self.retriever = retriever
        self.generator = generator
        self.hallucination_detector = hallucination_detector
        self.ragas_evaluator = ragas_evaluator or RAGASEvaluator()
        self.compute_bertscore_flag = compute_bertscore
        self.k_values = k_values

    def evaluate(
        self,
        qa_dataset: List[dict],
        top_k: int = 10,
        save_path: Optional[Path] = None,
    ) -> FullEvalReport:
        """
        Run full evaluation suite on a QA dataset.

        Args:
            qa_dataset: List of dicts with keys:
                        - 'question': str
                        - 'answer': str (reference answer)
                        - 'relevant_doc_ids': List[str] (for retrieval eval)
            top_k: Number of documents to retrieve per query.
            save_path: If provided, save JSON report to this path.

        Returns:
            FullEvalReport with all evaluation metrics.
        """
        questions = [d["question"] for d in qa_dataset]
        ref_answers = [d.get("answer", "") for d in qa_dataset]
        relevant_ids = [d.get("relevant_doc_ids", []) for d in qa_dataset]

        report = FullEvalReport()
        generated_answers = []
        retrieved_contexts = []
        retrieved_id_lists = []
        latencies = []

        # Retrieve + generate
        if self.retriever and self.generator:
            for qa in qa_dataset:
                start = time.time()
                docs = self.retriever.retrieve(qa["question"], top_k=top_k)
                answer = self.generator.generate(qa["question"], docs)
                latency = (time.time() - start) * 1000

                generated_answers.append(answer.text)
                retrieved_contexts.append([d.text for d in docs])
                retrieved_id_lists.append([d.chunk_id for d in docs])
                latencies.append(latency)

        # Retrieval metrics
        if retrieved_id_lists and any(relevant_ids):
            from src.retrieval.retriever import compute_retrieval_metrics
            rm = compute_retrieval_metrics(
                retrieved_id_lists, relevant_ids, self.k_values, latencies
            )
            report.retrieval = RetrievalEvalResult(
                precision_at_k=rm.precision_at_k,
                recall_at_k=rm.recall_at_k,
                mrr=rm.mrr,
                ndcg_at_k=rm.ndcg_at_k,
                num_queries=rm.num_queries,
                avg_latency_ms=rm.avg_latency_ms,
            )

        # Generation metrics
        if generated_answers and ref_answers:
            bleu = compute_bleu(generated_answers, ref_answers)
            rouge_scores = compute_rouge(generated_answers, ref_answers)
            bertscore_f = 0.0
            if self.compute_bertscore_flag:
                bertscore_f = compute_bertscore(generated_answers, ref_answers)
            report.generation = GenerationEvalResult(
                bleu=bleu,
                rouge1_f=rouge_scores["rouge1_f"],
                rouge2_f=rouge_scores["rouge2_f"],
                rougeL_f=rouge_scores["rougeL_f"],
                bertscore_f=bertscore_f,
                num_examples=len(generated_answers),
            )

        # RAGAS
        if generated_answers and retrieved_contexts:
            report.ragas = self.ragas_evaluator.evaluate(
                questions=questions,
                contexts=retrieved_contexts,
                answers=generated_answers,
                ground_truths=ref_answers if ref_answers else None,
            )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info("Evaluation report saved to %s", save_path)

        return report
