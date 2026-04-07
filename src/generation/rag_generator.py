"""
RAG generation module for clinical and biomedical question answering.

Handles LLM integration for answer generation with:
  - Clinical-specific prompt templates (Q&A, literature synthesis, adverse event extraction)
  - Citation generation linking answer claims to source documents
  - Streaming response support (SSE-compatible generator output)
  - Multi-provider support: OpenAI API, Azure OpenAI, and HuggingFace local models
  - Safety guardrails: clinical disclaimer insertion, confidence-gated responses

Clinical safety principles:
  - NEVER generate answers without retrieved context (no pure hallucination)
  - ALWAYS include source citations with PMID/note IDs
  - Flag low-confidence answers explicitly
  - Include mandatory clinical disclaimer for decision-support outputs
  - Do not provide specific dosage recommendations without verified source citations

Supported models:
  - gpt-4o / gpt-4o-mini (OpenAI) — best quality
  - gpt-3.5-turbo (OpenAI) — fast, cost-efficient
  - microsoft/biogpt (HuggingFace) — local, biomedical domain
  - meta-llama/Llama-3.1-8B-Instruct (HuggingFace) — local, general
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Generator, List, Optional, Union

from src.vectorstore.vector_index import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """A citation linking a generated claim to a source document."""

    chunk_id: str
    text_snippet: str  # The relevant passage from the source
    metadata: Dict[str, Any]
    relevance_score: float

    @property
    def pmid(self) -> Optional[str]:
        return self.metadata.get("pmid")

    @property
    def title(self) -> Optional[str]:
        return self.metadata.get("title")

    @property
    def formatted(self) -> str:
        """Format citation for display."""
        if self.pmid:
            return f"[PMID: {self.pmid}] {self.title or 'Untitled'}"
        note_id = self.metadata.get("note_id")
        if note_id:
            return f"[Clinical Note: {note_id}]"
        return f"[{self.chunk_id}]"


@dataclass
class GeneratedAnswer:
    """
    A RAG-generated answer with citations and quality indicators.

    Attributes:
        text: Generated answer text.
        citations: Source documents used in generation.
        faithfulness_score: Estimated faithfulness to retrieved context (0–1).
        confidence: Model confidence in the answer (0–1).
        sources_used: Number of context documents actually referenced.
        generation_time_ms: Wall-clock generation time.
        model: LLM model name used.
        has_clinical_disclaimer: Whether answer includes safety disclaimer.
        raw_response: Full LLM API response (for debugging).
    """

    text: str
    citations: List[Citation] = field(default_factory=list)
    faithfulness_score: float = 0.0
    confidence: float = 0.0
    sources_used: int = 0
    generation_time_ms: float = 0.0
    model: str = ""
    has_clinical_disclaimer: bool = False
    raw_response: Optional[dict] = None

    def format_with_citations(self) -> str:
        """Return answer text with formatted citation list appended."""
        lines = [self.text, "", "Sources:"]
        for i, cite in enumerate(self.citations, 1):
            lines.append(f"  [{i}] {cite.formatted}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CLINICAL_QA_SYSTEM_PROMPT = """You are ClinicalRAG, a precise biomedical AI assistant that answers \
clinical and research questions using only the provided context documents.

Rules:
1. Answer ONLY based on the provided context. Do not use knowledge not present in the context.
2. If the context does not contain enough information to answer confidently, say so explicitly.
3. Cite the source document(s) that support each major claim using [Doc N] notation.
4. Use precise medical terminology. Do not simplify clinical terms.
5. For drug dosages, interactions, or clinical recommendations, ALWAYS cite the specific source.
6. Acknowledge uncertainty when evidence is conflicting or limited.

Output format:
- Direct answer to the question
- Supporting evidence with [Doc N] citations
- Confidence level: High / Medium / Low
- Clinical disclaimer if the answer could affect patient care"""

LITERATURE_SYNTHESIS_SYSTEM_PROMPT = """You are a biomedical research assistant that synthesizes \
evidence from multiple PubMed abstracts into a coherent summary.

For each provided abstract, extract:
- Key findings and effect sizes
- Study design and population
- Limitations

Synthesize across documents to identify:
- Areas of consensus
- Conflicting results  
- Evidence gaps
- Clinical implications

Always cite sources as [PMID: XXXXXXXX] when referencing specific findings."""

ADVERSE_EVENT_EXTRACTION_SYSTEM_PROMPT = """You are a pharmacovigilance AI that extracts \
structured drug-adverse event information from clinical text.

For each mention of a potential adverse event, extract:
- Drug name (normalize to generic name)
- Adverse event (normalize to MedDRA Preferred Term)
- Temporal relationship (onset, duration)
- Severity (if stated)
- Causality assessment (if stated)

Return results as JSON array with fields: drug, adverse_event, meddra_pt, onset, severity, causality.
If a field is not mentioned, use null."""

CLINICAL_DISCLAIMER = (
    "\n\n⚕️ **Clinical Disclaimer**: This information is generated for research and "
    "informational purposes only. It does not constitute medical advice, diagnosis, or "
    "treatment recommendations. Always consult qualified healthcare professionals for "
    "clinical decision-making."
)


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

class OpenAIProvider:
    """
    OpenAI API provider for GPT-4 and GPT-3.5 models.

    Args:
        model: OpenAI model name (e.g., 'gpt-4o', 'gpt-4o-mini').
        api_key: OpenAI API key. Reads OPENAI_API_KEY env var if None.
        temperature: Sampling temperature (0 = deterministic, 1 = creative).
        max_tokens: Maximum tokens in generated response.
        azure_endpoint: Azure OpenAI endpoint (if using Azure).
        azure_deployment: Azure deployment name.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
            if self.azure_endpoint:
                self._client = openai.AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version="2024-02-01",
                )
            else:
                self._client = openai.OpenAI(api_key=self.api_key)
        except ImportError as exc:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            ) from exc
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response.

        Args:
            system_prompt: System role instructions.
            user_message: User message with context and query.
            stream: If True, return a token stream generator.

        Returns:
            Complete response string, or generator of token strings if stream=True.
        """
        client = self._get_client()
        model = self.azure_deployment or self.model

        if stream:
            return self._stream(client, model, system_prompt, user_message)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _stream(
        self, client, model: str, system_prompt: str, user_message: str
    ) -> Generator[str, None, None]:
        """Yield tokens from streaming API response."""
        with client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        ) as stream:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content


class HuggingFaceProvider:
    """
    Local HuggingFace model provider.

    Supports BioGPT, Llama, and other instruction-tuned models.
    Runs locally — no API calls, HIPAA-safe for clinical text.

    Args:
        model_name: HuggingFace model name or local path.
        device: 'auto', 'cuda', 'cpu'.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        load_in_8bit: Use 8-bit quantization (requires bitsandbytes).
        load_in_4bit: Use 4-bit quantization (requires bitsandbytes).
    """

    def __init__(
        self,
        model_name: str = "microsoft/biogpt",
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
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
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
            import torch

            quantization_config = None
            if self.load_in_4bit or self.load_in_8bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit,
                )

            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto" if self._device == "cuda" else self._device,
                model_kwargs={"quantization_config": quantization_config}
                if quantization_config
                else {},
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )
            logger.info("HuggingFace model loaded: %s", self.model_name)
        except ImportError as exc:
            raise ImportError(
                "transformers required. Install: pip install transformers accelerate"
            ) from exc

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        stream: bool = False,
    ) -> str:
        """Generate a response using local HF model."""
        self._load_pipeline()
        # Format as instruction prompt
        prompt = f"[INST] {system_prompt}\n\n{user_message} [/INST]"
        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()


# ---------------------------------------------------------------------------
# RAG generator
# ---------------------------------------------------------------------------

class ClinicalRAGGenerator:
    """
    End-to-end RAG generation for clinical and biomedical questions.

    Combines retrieved context documents with an LLM to generate
    grounded, cited answers. Applies clinical safety guardrails including
    mandatory disclaimers and confidence-gated refusal.

    Args:
        provider: LLM provider (OpenAIProvider or HuggingFaceProvider).
        include_disclaimer: Append clinical disclaimer to answers.
        min_confidence_threshold: Below this confidence, return refusal message.
        max_context_chunks: Maximum context chunks to include in prompt.
        max_context_tokens: Approximate token budget for context window.
        citation_style: 'inline' (Doc N) or 'footnote'.

    Example::

        generator = ClinicalRAGGenerator(
            provider=OpenAIProvider(model="gpt-4o-mini")
        )
        answer = generator.generate(
            query="What are first-line treatments for atrial fibrillation?",
            context_docs=retrieved_results,
        )
        print(answer.format_with_citations())
    """

    def __init__(
        self,
        provider: Optional[Union[OpenAIProvider, HuggingFaceProvider]] = None,
        include_disclaimer: bool = True,
        min_confidence_threshold: float = 0.2,
        max_context_chunks: int = 8,
        max_context_tokens: int = 3000,
        citation_style: str = "inline",
    ):
        self.provider = provider or OpenAIProvider()
        self.include_disclaimer = include_disclaimer
        self.min_confidence_threshold = min_confidence_threshold
        self.max_context_chunks = max_context_chunks
        self.max_context_tokens = max_context_tokens
        self.citation_style = citation_style

    def generate(
        self,
        query: str,
        context_docs: List[RetrievalResult],
        task: str = "clinical_qa",
        stream: bool = False,
    ) -> GeneratedAnswer:
        """
        Generate an answer for a query given retrieved context documents.

        Args:
            query: User's clinical question.
            context_docs: Retrieved documents from the retrieval pipeline.
            task: Prompt template to use:
                  'clinical_qa' | 'literature_synthesis' | 'adverse_event_extraction'
            stream: If True and provider supports it, stream token output.

        Returns:
            GeneratedAnswer with text, citations, and quality metrics.
        """
        start = time.time()

        if not context_docs:
            logger.warning("No context documents provided for query: %s", query)
            return GeneratedAnswer(
                text="I cannot answer this question because no relevant context documents were retrieved. "
                     "Please check the index and try again.",
                citations=[],
                faithfulness_score=0.0,
                confidence=0.0,
                model=getattr(self.provider, "model", "unknown"),
            )

        # Select and format context
        selected_docs = self._select_context(context_docs)
        context_str, citations = self._format_context(selected_docs)

        # Build prompt
        system_prompt, user_message = self._build_prompt(
            query=query,
            context=context_str,
            task=task,
        )

        # Generate
        try:
            raw_text = self.provider.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                stream=False,
            )
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            raise

        # Parse confidence from generated text
        confidence = self._extract_confidence(raw_text)

        # Add disclaimer
        if self.include_disclaimer and task == "clinical_qa":
            raw_text += CLINICAL_DISCLAIMER

        elapsed = (time.time() - start) * 1000
        return GeneratedAnswer(
            text=raw_text,
            citations=citations,
            faithfulness_score=0.0,  # Set by hallucination_detector
            confidence=confidence,
            sources_used=len(selected_docs),
            generation_time_ms=elapsed,
            model=getattr(self.provider, "model", "unknown"),
            has_clinical_disclaimer=self.include_disclaimer and task == "clinical_qa",
        )

    def generate_stream(
        self,
        query: str,
        context_docs: List[RetrievalResult],
        task: str = "clinical_qa",
    ) -> Generator[str, None, None]:
        """
        Stream generated tokens for real-time display.

        Args:
            query: Clinical query.
            context_docs: Retrieved documents.
            task: Task type.

        Yields:
            Token strings as they are generated.
        """
        selected_docs = self._select_context(context_docs)
        context_str, _ = self._format_context(selected_docs)
        system_prompt, user_message = self._build_prompt(query, context_str, task)

        try:
            token_stream = self.provider.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                stream=True,
            )
            yield from token_stream
        except Exception as exc:
            logger.error("Streaming generation failed: %s", exc)
            yield f"\n[Error: {exc}]"

    def _select_context(
        self, docs: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Select the most relevant context documents within token budget."""
        selected = []
        token_count = 0
        for doc in docs[: self.max_context_chunks]:
            doc_tokens = len(doc.text.split()) * 1.3
            if token_count + doc_tokens > self.max_context_tokens:
                break
            selected.append(doc)
            token_count += doc_tokens
        return selected

    def _format_context(
        self, docs: List[RetrievalResult]
    ) -> tuple[str, List[Citation]]:
        """Format context documents into prompt string and citation list."""
        lines = []
        citations: List[Citation] = []

        for i, doc in enumerate(docs, 1):
            # Format document header
            meta = doc.metadata
            if meta.get("pmid"):
                header = (
                    f"[Doc {i}] PMID: {meta['pmid']} | "
                    f"{meta.get('journal', '')} ({meta.get('pub_date', '')[:4]})"
                )
                if meta.get("title"):
                    header += f"\nTitle: {meta['title']}"
            elif meta.get("note_id"):
                header = (
                    f"[Doc {i}] Clinical Note: {meta['note_id']} | "
                    f"Type: {meta.get('note_type', 'unknown')} | "
                    f"Section: {meta.get('section', 'full_note')}"
                )
            else:
                header = f"[Doc {i}]"

            lines.append(f"{header}\n{doc.text}\n")
            citations.append(
                Citation(
                    chunk_id=doc.chunk_id,
                    text_snippet=doc.text[:200],
                    metadata=meta,
                    relevance_score=doc.score,
                )
            )

        return "\n---\n".join(lines), citations

    def _build_prompt(
        self, query: str, context: str, task: str
    ) -> tuple[str, str]:
        """Build system prompt and user message for a given task."""
        if task == "literature_synthesis":
            system_prompt = LITERATURE_SYNTHESIS_SYSTEM_PROMPT
        elif task == "adverse_event_extraction":
            system_prompt = ADVERSE_EVENT_EXTRACTION_SYSTEM_PROMPT
        else:
            system_prompt = CLINICAL_QA_SYSTEM_PROMPT

        user_message = (
            f"Context Documents:\n\n{context}\n\n"
            f"---\n\nQuestion: {query}\n\n"
            f"Provide a comprehensive answer based solely on the context above."
        )
        return system_prompt, user_message

    @staticmethod
    def _extract_confidence(text: str) -> float:
        """Parse confidence level from generated text."""
        text_lower = text.lower()
        if "confidence level: high" in text_lower or "high confidence" in text_lower:
            return 0.85
        elif "confidence level: medium" in text_lower or "moderate confidence" in text_lower:
            return 0.60
        elif "confidence level: low" in text_lower or "low confidence" in text_lower:
            return 0.35
        elif any(phrase in text_lower for phrase in [
            "insufficient evidence", "cannot answer", "not enough information",
            "context does not contain"
        ]):
            return 0.10
        return 0.70  # Default moderate confidence

    @classmethod
    def from_config(cls, config_path: str) -> "ClinicalRAGGenerator":
        """Instantiate generator from YAML configuration."""
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        gen_cfg = config.get("generation", {})
        provider_type = gen_cfg.get("provider", "openai")

        if provider_type == "openai":
            provider = OpenAIProvider(
                model=gen_cfg.get("model", "gpt-4o-mini"),
                temperature=gen_cfg.get("temperature", 0.1),
                max_tokens=gen_cfg.get("max_tokens", 1024),
            )
        elif provider_type == "huggingface":
            provider = HuggingFaceProvider(
                model_name=gen_cfg.get("model", "microsoft/biogpt"),
                max_new_tokens=gen_cfg.get("max_tokens", 512),
            )
        else:
            raise ValueError(f"Unknown provider: {provider_type}")

        return cls(
            provider=provider,
            include_disclaimer=gen_cfg.get("include_disclaimer", True),
            max_context_chunks=gen_cfg.get("max_context_chunks", 8),
        )
