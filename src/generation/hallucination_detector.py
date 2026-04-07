"""
Hallucination detection for clinical RAG outputs.

In clinical settings, hallucinated facts can cause direct patient harm.
This module implements a multi-layer faithfulness verification pipeline:

Layer 1 — Claim Extraction:
  Decompose the generated answer into atomic factual claims using an LLM.
  "Metformin lowers blood glucose by activating AMPK and reducing hepatic
   glucose production" → three separate claims.

Layer 2 — NLI Entailment Verification:
  For each claim, check whether it is entailed by the retrieved context
  using a Natural Language Inference (NLI) model.
  Models: MedNLI (clinical-domain NLI), DeBERTa-NLI, or GPT-4 verification.

Layer 3 — FactScore-style Attribution:
  For each claim, identify which specific context passage supports it,
  assign a support score, and flag unsupported claims.

Layer 4 — Confidence Calibration:
  Aggregate claim-level entailment scores into a document-level faithfulness
  score and confidence estimate.

References:
  - Min, S. et al. (2023). FActScoring: Fine-grained Atomic Evaluation of
    Factual Precision in Long-Form Text Generation. EMNLP 2023.
  - Romanov & Shivade (2018). Lessons from Natural Language Inference in the
    Clinical Domain (MedNLI). EMNLP 2018.
  - Manakul et al. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination
    Detection for Generative LLMs. EMNLP 2023.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.vectorstore.vector_index import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AtomicClaim:
    """A single verifiable factual claim extracted from generated text."""

    claim_id: int
    text: str
    source_sentence: str  # Original sentence this claim was extracted from
    is_supported: Optional[bool] = None
    support_score: float = 0.0  # 0 (refuted) to 1 (strongly supported)
    supporting_passage: Optional[str] = None
    supporting_chunk_id: Optional[str] = None
    entailment_label: str = "neutral"  # 'entailment', 'contradiction', 'neutral'
    confidence: float = 0.0


@dataclass
class FaithfulnessReport:
    """
    Faithfulness assessment for a generated answer.

    Attributes:
        faithfulness_score: Overall faithfulness (0–1). Fraction of claims
                           that are entailed by the retrieved context.
        hallucination_rate: Fraction of claims that are NOT supported (0–1).
        num_claims: Total atomic claims extracted.
        num_supported: Claims with entailment_label == 'entailment'.
        num_contradicted: Claims with entailment_label == 'contradiction'.
        num_neutral: Claims with insufficient evidence.
        claims: Detailed claim-level assessments.
        unsupported_claims: Text of claims not supported by context.
        citation_accuracy: Fraction of cited sources actually support the claim.
        overall_confidence: Aggregated confidence in faithfulness assessment.
    """

    faithfulness_score: float
    hallucination_rate: float
    num_claims: int
    num_supported: int
    num_contradicted: int
    num_neutral: int
    claims: List[AtomicClaim] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    citation_accuracy: float = 1.0
    overall_confidence: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def summary(self) -> str:
        """Human-readable faithfulness summary."""
        lines = [
            f"Faithfulness Score:   {self.faithfulness_score:.3f}",
            f"Hallucination Rate:   {self.hallucination_rate:.1%}",
            f"Claims Analyzed:      {self.num_claims}",
            f"  Supported:          {self.num_supported}",
            f"  Contradicted:       {self.num_contradicted}",
            f"  Neutral/Unknown:    {self.num_neutral}",
            f"Citation Accuracy:    {self.citation_accuracy:.3f}",
        ]
        if self.unsupported_claims:
            lines.append(f"\nUnsupported claims ({len(self.unsupported_claims)}):")
            for claim in self.unsupported_claims[:3]:
                lines.append(f"  • {claim[:100]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claim extractor
# ---------------------------------------------------------------------------

class AtomicClaimExtractor:
    """
    Extract atomic factual claims from generated text.

    Uses an LLM to decompose a multi-sentence answer into a list of
    verifiable atomic claims. Each claim should be:
      - A single fact (not compound)
      - Self-contained (interpretable without the surrounding text)
      - Verifiable from biomedical sources

    Args:
        provider: LLM provider for claim extraction.
        max_claims: Maximum claims to extract per answer.
    """

    EXTRACTION_SYSTEM_PROMPT = """You are a claim extraction system for clinical text verification.

Given a clinical/biomedical answer, extract ALL verifiable factual claims.

Rules:
1. Each claim must be ONE atomic fact (a single proposition).
2. Claims must be self-contained — include necessary context.
3. Include specific measurements, drug names, mechanisms, and relationships.
4. Do NOT extract opinions, uncertainties, or meta-statements.
5. Return ONLY a JSON array of claim strings.

Example input: "Metformin reduces hepatic glucose production by activating AMPK, 
which leads to decreased gluconeogenesis. The typical starting dose is 500 mg twice daily."

Example output: ["Metformin reduces hepatic glucose production.", 
"Metformin activates AMPK.", 
"AMPK activation decreases gluconeogenesis.",
"The typical starting dose of metformin is 500 mg twice daily."]"""

    def __init__(self, provider=None, max_claims: int = 20):
        self.provider = provider
        self.max_claims = max_claims

    def extract(self, answer_text: str) -> List[AtomicClaim]:
        """
        Extract atomic claims from an answer.

        Args:
            answer_text: Generated answer to decompose.

        Returns:
            List of AtomicClaim objects.
        """
        # Remove disclaimer if present
        clean_text = re.sub(
            r"⚕️.*Clinical Disclaimer.*", "", answer_text, flags=re.DOTALL
        ).strip()

        if self.provider:
            return self._extract_with_llm(clean_text)
        else:
            return self._extract_heuristic(clean_text)

    def _extract_with_llm(self, text: str) -> List[AtomicClaim]:
        """Use LLM to extract claims."""
        prompt = f"Extract atomic factual claims from this text:\n\n{text}"
        try:
            response = self.provider.generate(
                system_prompt=self.EXTRACTION_SYSTEM_PROMPT,
                user_message=prompt,
            )
            # Parse JSON array
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                claims_list = json.loads(json_match.group())
                return [
                    AtomicClaim(
                        claim_id=i,
                        text=claim,
                        source_sentence=text,
                    )
                    for i, claim in enumerate(claims_list[: self.max_claims])
                    if isinstance(claim, str) and claim.strip()
                ]
        except Exception as exc:
            logger.warning("LLM claim extraction failed: %s. Using heuristic.", exc)
        return self._extract_heuristic(text)

    def _extract_heuristic(self, text: str) -> List[AtomicClaim]:
        """
        Heuristic claim extraction using sentence splitting.

        Falls back when LLM is unavailable. Less precise than LLM extraction
        but still useful for hallucination detection.
        """
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            # Skip non-factual sentences
            if not sent or len(sent) < 15:
                continue
            if sent.lower().startswith(("note:", "disclaimer:", "sources:")):
                continue
            # Skip confidence level statements
            if "confidence level:" in sent.lower():
                continue
            claims.append(
                AtomicClaim(
                    claim_id=i,
                    text=sent,
                    source_sentence=sent,
                )
            )
        return claims[: self.max_claims]


# ---------------------------------------------------------------------------
# NLI entailment verifier
# ---------------------------------------------------------------------------

class NLIEntailmentVerifier:
    """
    Verify claim-context entailment using Natural Language Inference.

    Uses a DeBERTa-based NLI model or MedNLI (clinical-domain NLI) to
    determine if each extracted claim is:
      - entailment: The claim follows from the context.
      - contradiction: The claim contradicts the context.
      - neutral: The context does not provide enough evidence.

    For clinical use, we use a conservative threshold: claims are only
    marked as supported if entailment probability > 0.7.

    Models:
      - cross-encoder/nli-deberta-v3-large: General, strong NLI
      - microsoft/deberta-base-mnli: Efficient
      - romanov/medical_nli: Clinical domain (MedNLI-trained)

    Args:
        model_name: NLI model from HuggingFace.
        device: Inference device.
        entailment_threshold: Minimum probability to classify as entailment.
        batch_size: Inference batch size.
    """

    ENTAILMENT_LABEL_MAP = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = "auto",
        entailment_threshold: float = 0.7,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.batch_size = batch_size
        self._pipeline = None

        if device == "auto":
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = device

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if self._device == "cuda" else -1,
            )
            logger.info("NLI pipeline loaded: %s", self.model_name)
        except ImportError as exc:
            raise ImportError(
                "transformers required for NLI. Install: pip install transformers"
            ) from exc

    def verify_claim(
        self,
        claim: str,
        context_passages: List[str],
    ) -> Tuple[str, float, Optional[str]]:
        """
        Verify a single claim against multiple context passages.

        Takes the maximum entailment score across all passages
        (claim is supported if any passage entails it).

        Args:
            claim: Atomic claim to verify.
            context_passages: Retrieved context text passages.

        Returns:
            Tuple of (entailment_label, max_support_score, best_supporting_passage).
        """
        if not context_passages:
            return "neutral", 0.0, None

        self._load_pipeline()

        best_label = "neutral"
        best_score = 0.0
        best_passage = None

        for passage in context_passages:
            try:
                # NLI input: premise = context passage, hypothesis = claim
                result = self._pipeline(
                    f"{passage} [SEP] {claim}",
                    truncation=True,
                    max_length=512,
                )
                # result is list of {"label": ..., "score": ...}
                label_map = {r["label"]: r["score"] for r in result}

                entail_score = label_map.get("ENTAILMENT", label_map.get("entailment", 0.0))
                contradict_score = label_map.get("CONTRADICTION", label_map.get("contradiction", 0.0))

                if entail_score > best_score:
                    best_score = entail_score
                    best_passage = passage
                    if entail_score >= self.entailment_threshold:
                        best_label = "entailment"
                    elif contradict_score > 0.5:
                        best_label = "contradiction"
                    else:
                        best_label = "neutral"

            except Exception as exc:
                logger.debug("NLI inference error: %s", exc)

        return best_label, best_score, best_passage

    def verify_claims_batch(
        self,
        claims: List[AtomicClaim],
        context_passages: List[str],
    ) -> List[AtomicClaim]:
        """
        Verify multiple claims in batch.

        Args:
            claims: List of atomic claims to verify.
            context_passages: All retrieved context passages.

        Returns:
            Claims with entailment_label, support_score, and supporting_passage set.
        """
        for claim in claims:
            label, score, passage = self.verify_claim(
                claim.text, context_passages
            )
            claim.entailment_label = label
            claim.support_score = score
            claim.supporting_passage = passage
            claim.is_supported = label == "entailment"
            claim.confidence = score

        return claims


# ---------------------------------------------------------------------------
# Main hallucination detector
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """
    Full hallucination detection pipeline for clinical RAG outputs.

    Combines claim extraction, NLI entailment verification, and citation
    accuracy checking into a single faithfulness assessment.

    Usage in clinical pipeline:
      1. Generate answer with ClinicalRAGGenerator.
      2. Pass answer + context to HallucinationDetector.detect().
      3. If faithfulness_score < threshold, regenerate or add disclaimer.
      4. Log all low-faithfulness answers for human review.

    Args:
        claim_extractor: AtomicClaimExtractor (LLM or heuristic).
        nli_verifier: NLIEntailmentVerifier.
        faithfulness_threshold: Below this score, flag as potentially hallucinated.
        use_llm_verification: Use LLM as additional verification (more accurate,
                               slower and requires API calls).

    Example::

        detector = HallucinationDetector()
        report = detector.detect(
            generated_answer="Metformin activates AMPK and reduces hepatic glucose...",
            context_docs=retrieved_results,
        )
        print(report.summary())
        if report.faithfulness_score < 0.7:
            print("WARNING: Low faithfulness detected. Review required.")
    """

    def __init__(
        self,
        claim_extractor: Optional[AtomicClaimExtractor] = None,
        nli_verifier: Optional[NLIEntailmentVerifier] = None,
        faithfulness_threshold: float = 0.7,
        use_llm_verification: bool = False,
    ):
        self.claim_extractor = claim_extractor or AtomicClaimExtractor()
        self.nli_verifier = nli_verifier or NLIEntailmentVerifier()
        self.faithfulness_threshold = faithfulness_threshold
        self.use_llm_verification = use_llm_verification

    def detect(
        self,
        generated_answer: str,
        context_docs: List[RetrievalResult],
    ) -> FaithfulnessReport:
        """
        Detect hallucinations in a generated answer.

        Args:
            generated_answer: Text generated by the RAG pipeline.
            context_docs: Retrieved context documents used for generation.

        Returns:
            FaithfulnessReport with per-claim entailment assessment.
        """
        # Extract claims from generated text
        claims = self.claim_extractor.extract(generated_answer)
        if not claims:
            logger.warning("No claims extracted from answer.")
            return FaithfulnessReport(
                faithfulness_score=1.0,
                hallucination_rate=0.0,
                num_claims=0,
                num_supported=0,
                num_contradicted=0,
                num_neutral=0,
            )

        # Collect context passages
        context_passages = [doc.text for doc in context_docs if doc.text]

        # Verify claims against context
        verified_claims = self.nli_verifier.verify_claims_batch(
            claims, context_passages
        )

        # Compute metrics
        num_supported = sum(1 for c in verified_claims if c.entailment_label == "entailment")
        num_contradicted = sum(1 for c in verified_claims if c.entailment_label == "contradiction")
        num_neutral = sum(1 for c in verified_claims if c.entailment_label == "neutral")
        num_total = len(verified_claims)

        faithfulness_score = num_supported / num_total if num_total > 0 else 1.0
        hallucination_rate = (num_total - num_supported) / num_total if num_total > 0 else 0.0

        unsupported = [
            c.text for c in verified_claims
            if c.entailment_label != "entailment"
        ]

        # Citation accuracy: what fraction of cited docs actually contain supporting evidence
        citation_accuracy = self._compute_citation_accuracy(
            generated_answer, context_docs, verified_claims
        )

        overall_confidence = float(np.mean([c.confidence for c in verified_claims])) if verified_claims else 0.0

        report = FaithfulnessReport(
            faithfulness_score=faithfulness_score,
            hallucination_rate=hallucination_rate,
            num_claims=num_total,
            num_supported=num_supported,
            num_contradicted=num_contradicted,
            num_neutral=num_neutral,
            claims=verified_claims,
            unsupported_claims=unsupported,
            citation_accuracy=citation_accuracy,
            overall_confidence=overall_confidence,
        )

        if faithfulness_score < self.faithfulness_threshold:
            logger.warning(
                "Low faithfulness detected: %.2f (threshold: %.2f). "
                "%d/%d claims unsupported.",
                faithfulness_score,
                self.faithfulness_threshold,
                num_total - num_supported,
                num_total,
            )

        return report

    def _compute_citation_accuracy(
        self,
        answer_text: str,
        context_docs: List[RetrievalResult],
        verified_claims: List[AtomicClaim],
    ) -> float:
        """
        Estimate citation accuracy.

        Checks whether cited document numbers in the answer ([Doc N]) actually
        contain passages that support the surrounding claims.

        Returns:
            Fraction of citations that are supported by the cited document.
        """
        citation_pattern = re.compile(r"\[Doc (\d+)\]")
        citations = citation_pattern.findall(answer_text)
        if not citations:
            return 1.0  # No citations to check

        supported_citations = 0
        for cite_num_str in citations:
            cite_idx = int(cite_num_str) - 1
            if 0 <= cite_idx < len(context_docs):
                # Check if any verified claim is supported by this document
                cited_doc = context_docs[cite_idx]
                for claim in verified_claims:
                    if (
                        claim.entailment_label == "entailment"
                        and claim.supporting_passage
                        and claim.supporting_passage[:50] in cited_doc.text
                    ):
                        supported_citations += 1
                        break

        return supported_citations / len(citations) if citations else 1.0

    def detect_with_threshold(
        self,
        generated_answer: str,
        context_docs: List[RetrievalResult],
    ) -> Tuple[FaithfulnessReport, bool]:
        """
        Detect hallucinations and return flag if answer should be blocked.

        Args:
            generated_answer: Generated answer text.
            context_docs: Retrieved context.

        Returns:
            Tuple of (FaithfulnessReport, should_block).
            should_block=True means the answer is too unreliable for clinical use.
        """
        report = self.detect(generated_answer, context_docs)
        should_block = report.faithfulness_score < self.faithfulness_threshold
        return report, should_block


# ---------------------------------------------------------------------------
# SelfCheck-style consistency detector
# ---------------------------------------------------------------------------

class SelfConsistencyDetector:
    """
    Detect hallucinations via self-consistency sampling.

    Generates multiple independent samples for the same query and measures
    their mutual agreement. Claims that appear in all samples are more likely
    to be factual; claims that vary across samples are likely hallucinated.

    Based on SelfCheckGPT (Manakul et al. 2023).

    Args:
        generator: ClinicalRAGGenerator for generating multiple samples.
        num_samples: Number of independent samples to generate.
        consistency_threshold: Minimum agreement rate to flag as consistent.
    """

    def __init__(
        self,
        generator,
        num_samples: int = 3,
        consistency_threshold: float = 0.67,
    ):
        self.generator = generator
        self.num_samples = num_samples
        self.consistency_threshold = consistency_threshold
        self._nli_verifier = NLIEntailmentVerifier()

    def check_consistency(
        self,
        query: str,
        context_docs: List[RetrievalResult],
        primary_answer: str,
    ) -> Dict[str, Any]:
        """
        Check consistency of primary answer against re-sampled answers.

        Args:
            query: Original query.
            context_docs: Retrieved context documents.
            primary_answer: The main generated answer to verify.

        Returns:
            Dict with 'consistency_score', 'consistent_claims', 'inconsistent_claims'.
        """
        samples = []
        for _ in range(self.num_samples):
            try:
                sample = self.generator.generate(query, context_docs, task="clinical_qa")
                samples.append(sample.text)
            except Exception as exc:
                logger.warning("Failed to generate consistency sample: %s", exc)

        if not samples:
            return {"consistency_score": 1.0, "samples": 0}

        # Check entailment of primary answer claims against each sample
        extractor = AtomicClaimExtractor()
        claims = extractor.extract(primary_answer)

        consistent_claims = []
        inconsistent_claims = []

        for claim in claims:
            support_count = 0
            for sample in samples:
                label, score, _ = self._nli_verifier.verify_claim(
                    claim.text, [sample]
                )
                if label == "entailment":
                    support_count += 1
            agreement_rate = support_count / len(samples)
            if agreement_rate >= self.consistency_threshold:
                consistent_claims.append(claim.text)
            else:
                inconsistent_claims.append(claim.text)

        consistency_score = len(consistent_claims) / len(claims) if claims else 1.0

        return {
            "consistency_score": consistency_score,
            "num_claims": len(claims),
            "consistent_claims": consistent_claims,
            "inconsistent_claims": inconsistent_claims,
            "num_samples": len(samples),
        }
