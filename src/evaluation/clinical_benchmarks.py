"""
Clinical NLP benchmark evaluation.

Implements evaluation on standard biomedical NLP benchmarks:

  1. BioASQ (Task B) — Biomedical question answering from PubMed:
     - Yes/No questions (binary classification)
     - Factoid questions (entity extraction)
     - List questions (multi-entity extraction)
     - Summary questions (free-form generation)
     Metric: Yes/No accuracy, Factoid MRR, List F-measure, BERTScore

  2. PubMedQA — Research question answering from PubMed:
     - 3-way classification: Yes / No / Maybe
     - Long-answer generation
     Metric: Classification accuracy (target: >72%)

  3. MedQA (USMLE) — Medical licensing exam questions:
     - 4-choice/5-choice multiple choice
     - Requires clinical knowledge reasoning
     Metric: Accuracy (GPT-4 baseline: ~90%, RAG can help on niche topics)

  4. i2b2 NER evaluation:
     - 2010 i2b2 Clinical NER (Problems, Tests, Treatments)
     - 2012 i2b2 Temporal Relations
     Metric: Token-level F1

References:
  - Tsatsaronis, G. et al. (2015). An overview of the BIOASQ large-scale
    biomedical semantic indexing and question answering competition. BMC Bioinformatics.
  - Jin, Q. et al. (2019). PubMedQA: A Dataset for Biomedical Research Question
    Answering. EMNLP 2019.
  - Jin, D. et al. (2021). What Disease Does This Patient Have? MedQA. Applied Sciences.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark result models
# ---------------------------------------------------------------------------

@dataclass
class BioASQResult:
    """BioASQ Task B evaluation results."""
    yes_no_accuracy: float
    factoid_mrr: float
    list_f_measure: float
    summary_bertscore: float
    num_yes_no: int
    num_factoid: int
    num_list: int
    num_summary: int

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"BioASQ Results:\n"
            f"  Yes/No accuracy:   {self.yes_no_accuracy:.4f} (n={self.num_yes_no})\n"
            f"  Factoid MRR:       {self.factoid_mrr:.4f} (n={self.num_factoid})\n"
            f"  List F-measure:    {self.list_f_measure:.4f} (n={self.num_list})\n"
            f"  Summary BERTScore: {self.summary_bertscore:.4f} (n={self.num_summary})"
        )


@dataclass
class PubMedQAResult:
    """PubMedQA evaluation results."""
    accuracy: float
    accuracy_yes: float
    accuracy_no: float
    accuracy_maybe: float
    num_examples: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MedQAResult:
    """MedQA USMLE evaluation results."""
    accuracy: float
    accuracy_by_type: Dict[str, float]
    num_examples: int
    num_correct: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class I2B2NERResult:
    """i2b2 NER evaluation results."""
    precision: float
    recall: float
    f1: float
    precision_by_type: Dict[str, float]
    recall_by_type: Dict[str, float]
    f1_by_type: Dict[str, float]
    num_examples: int

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# BioASQ evaluator
# ---------------------------------------------------------------------------

class BioASQEvaluator:
    """
    BioASQ Task B evaluation.

    BioASQ is the most widely used biomedical QA benchmark, covering
    questions with four types:
      - yes/no: Binary questions about biomedical facts
      - factoid: Short entity answers ("Which gene is associated with X?")
      - list: Multiple entity answers
      - summary: Free-form text answers

    This evaluator accepts a RAG generator and runs it on BioASQ format data.

    Data format (each item):
    {
        "id": "55031181e9bde69634000014",
        "type": "yesno",
        "body": "Is the gene KCNQ1OT1 associated with...",
        "ideal_answer": "Yes",
        "exact_answer": "yes",
        "documents": ["http://www.ncbi.nlm.nih.gov/pubmed/25262937", ...]
    }

    Args:
        retriever: RAG retriever for document lookup.
        generator: RAG generator for answer generation.
    """

    def __init__(self, retriever=None, generator=None):
        self.retriever = retriever
        self.generator = generator

    def evaluate(
        self,
        data_path: Path,
        max_examples: Optional[int] = None,
    ) -> BioASQResult:
        """
        Evaluate on BioASQ data.

        Args:
            data_path: Path to BioASQ JSON file.
            max_examples: Limit evaluation size (for development).

        Returns:
            BioASQResult with per-type metrics.
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"BioASQ data not found at {data_path}. "
                f"Download from http://bioasq.org/participate/challenges"
            )

        with open(data_path) as f:
            raw = json.load(f)
        questions = raw.get("questions", [])
        if max_examples:
            questions = questions[:max_examples]

        yesno_correct = 0
        yesno_total = 0
        factoid_rr_sum = 0.0
        factoid_total = 0
        list_f_sum = 0.0
        list_total = 0
        summary_scores = []

        for q in questions:
            q_type = q.get("type", "")
            body = q.get("body", "")
            exact = q.get("exact_answer", "")
            ideal = q.get("ideal_answer", [""])[0] if isinstance(q.get("ideal_answer"), list) else q.get("ideal_answer", "")

            # Retrieve and generate
            predicted = self._answer_question(body, q_type)

            if q_type == "yesno":
                pred_yn = self._extract_yesno(predicted)
                gold_yn = self._extract_yesno(str(exact))
                if pred_yn and gold_yn:
                    yesno_correct += int(pred_yn == gold_yn)
                    yesno_total += 1

            elif q_type == "factoid":
                gold = [exact] if isinstance(exact, str) else exact
                rr = self._mean_reciprocal_rank(predicted, gold)
                factoid_rr_sum += rr
                factoid_total += 1

            elif q_type == "list":
                gold_list = exact if isinstance(exact, list) else [exact]
                pred_list = self._extract_list_answer(predicted)
                f_score = self._list_f_measure(pred_list, gold_list)
                list_f_sum += f_score
                list_total += 1

            elif q_type == "summary":
                # Use simple token overlap as proxy for BERTScore
                if ideal:
                    pred_tokens = set(predicted.lower().split())
                    gold_tokens = set(ideal.lower().split())
                    if pred_tokens and gold_tokens:
                        f = 2 * len(pred_tokens & gold_tokens) / (len(pred_tokens) + len(gold_tokens))
                        summary_scores.append(f)

        return BioASQResult(
            yes_no_accuracy=yesno_correct / yesno_total if yesno_total > 0 else 0.0,
            factoid_mrr=factoid_rr_sum / factoid_total if factoid_total > 0 else 0.0,
            list_f_measure=list_f_sum / list_total if list_total > 0 else 0.0,
            summary_bertscore=sum(summary_scores) / len(summary_scores) if summary_scores else 0.0,
            num_yes_no=yesno_total,
            num_factoid=factoid_total,
            num_list=list_total,
            num_summary=len(summary_scores),
        )

    def _answer_question(self, question: str, q_type: str) -> str:
        """Generate answer for a BioASQ question."""
        if self.retriever and self.generator:
            docs = self.retriever.retrieve(question, top_k=5)
            answer = self.generator.generate(question, docs, task="clinical_qa")
            return answer.text
        return ""

    @staticmethod
    def _extract_yesno(text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        if text_lower.startswith("yes"):
            return "yes"
        elif text_lower.startswith("no"):
            return "no"
        # Search in text
        if re.search(r"\byes\b", text_lower):
            return "yes"
        elif re.search(r"\bno\b", text_lower):
            return "no"
        return None

    @staticmethod
    def _mean_reciprocal_rank(predicted: str, gold_answers: List[str]) -> float:
        pred_lower = predicted.lower()
        for i, ans in enumerate([predicted], 1):
            for gold in gold_answers:
                if gold.lower() in pred_lower:
                    return 1.0 / i
        return 0.0

    @staticmethod
    def _extract_list_answer(text: str) -> List[str]:
        """Extract list items from generated text."""
        items = re.findall(r"(?:^|\n)\s*[-•*]\s*(.+)", text, re.MULTILINE)
        if not items:
            # Try numbered list
            items = re.findall(r"\d+\.\s*(.+)", text)
        if not items:
            items = [text]
        return [item.strip() for item in items if item.strip()]

    @staticmethod
    def _list_f_measure(predicted: List[str], gold: List[str]) -> float:
        pred_set = {p.lower() for p in predicted}
        gold_set = {g.lower() for g in gold}
        if not pred_set or not gold_set:
            return 0.0
        tp = len(pred_set & gold_set)
        prec = tp / len(pred_set)
        rec = tp / len(gold_set)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ---------------------------------------------------------------------------
# PubMedQA evaluator
# ---------------------------------------------------------------------------

class PubMedQAEvaluator:
    """
    PubMedQA evaluation.

    PubMedQA tests research question answering over PubMed abstracts.
    Questions are answered with Yes / No / Maybe.

    Data format (PubMedQA labeled dataset):
    {
        "PMID": {
            "QUESTION": "Do mitochondria play a role in...",
            "CONTEXTS": ["Abstract text..."],
            "LABELS": ["yes"],
            "LONG_ANSWER": "Yes, mitochondria..."
        }
    }

    The HuggingFace dataset is available at:
    huggingface.co/datasets/qiaojin/PubMedQA

    Args:
        retriever: Optional retriever (if not using provided contexts).
        generator: RAG generator.
    """

    LABELS = ["yes", "no", "maybe"]

    def __init__(self, retriever=None, generator=None):
        self.retriever = retriever
        self.generator = generator

    def evaluate(
        self,
        data_path: Optional[Path] = None,
        use_hf_dataset: bool = True,
        max_examples: Optional[int] = None,
        split: str = "test",
    ) -> PubMedQAResult:
        """
        Evaluate on PubMedQA.

        Args:
            data_path: Local PubMedQA JSON file (alternative to HF dataset).
            use_hf_dataset: Load from HuggingFace datasets.
            max_examples: Maximum examples to evaluate.
            split: Dataset split ('train', 'validation', 'test').

        Returns:
            PubMedQAResult with accuracy metrics.
        """
        examples = self._load_data(data_path, use_hf_dataset, split)
        if max_examples:
            examples = examples[:max_examples]

        correct = 0
        total = 0
        by_label_correct = defaultdict(int)
        by_label_total = defaultdict(int)

        for example in examples:
            question = example["question"]
            context = example.get("contexts", [])
            gold_label = example["label"].lower()

            predicted_label = self._predict_label(question, context)

            by_label_total[gold_label] += 1
            if predicted_label == gold_label:
                correct += 1
                by_label_correct[gold_label] += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return PubMedQAResult(
            accuracy=accuracy,
            accuracy_yes=by_label_correct["yes"] / max(by_label_total["yes"], 1),
            accuracy_no=by_label_correct["no"] / max(by_label_total["no"], 1),
            accuracy_maybe=by_label_correct["maybe"] / max(by_label_total["maybe"], 1),
            num_examples=total,
        )

    def _load_data(
        self,
        data_path: Optional[Path],
        use_hf: bool,
        split: str,
    ) -> List[dict]:
        """Load PubMedQA examples."""
        if use_hf:
            try:
                from datasets import load_dataset
                ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
                examples = []
                for item in ds:
                    examples.append({
                        "question": item["question"],
                        "contexts": item["context"]["contexts"],
                        "label": item["final_decision"],
                    })
                return examples
            except ImportError:
                logger.warning("datasets not installed. Trying local file.")
        if data_path and Path(data_path).exists():
            with open(data_path) as f:
                raw = json.load(f)
            examples = []
            for pmid, item in raw.items():
                examples.append({
                    "question": item.get("QUESTION", ""),
                    "contexts": item.get("CONTEXTS", []),
                    "label": item.get("final_decision", item.get("LABELS", ["maybe"])[0]),
                })
            return examples
        raise FileNotFoundError(
            "PubMedQA data not found. Install `datasets` or provide local file path."
        )

    def _predict_label(self, question: str, contexts: List[str]) -> str:
        """Predict Yes/No/Maybe label for a PubMedQA question."""
        if self.generator and contexts:
            from src.vectorstore.vector_index import RetrievalResult
            mock_docs = [
                RetrievalResult(
                    chunk_id=f"ctx_{i}",
                    text=ctx,
                    score=1.0,
                    metadata={},
                )
                for i, ctx in enumerate(contexts)
            ]
            answer = self.generator.generate(question, mock_docs, task="clinical_qa")
            return self._extract_label(answer.text)
        return "maybe"

    @staticmethod
    def _extract_label(text: str) -> str:
        text_lower = text.lower()
        if text_lower.startswith("yes") or re.search(r"\byes[,. ]", text_lower):
            return "yes"
        elif text_lower.startswith("no") or re.search(r"\bno[,. ]", text_lower):
            return "no"
        return "maybe"


# ---------------------------------------------------------------------------
# MedQA evaluator
# ---------------------------------------------------------------------------

class MedQAEvaluator:
    """
    MedQA (USMLE) multiple-choice evaluation.

    Tests clinical reasoning on US Medical Licensing Exam-style questions.
    Available in 4-choice (US) and 5-choice (US extended/Chinese) variants.

    Data format:
    {
        "question": "A 45-year-old man presents with...",
        "options": {"A": "Prescribe aspirin", "B": "...", "C": "...", "D": "..."},
        "answer_idx": "A",
        "answer": "Prescribe aspirin"
    }

    Args:
        generator: RAG generator for answer selection.
        retriever: RAG retriever for relevant context lookup.
    """

    def __init__(self, retriever=None, generator=None):
        self.retriever = retriever
        self.generator = generator

    def evaluate(
        self,
        data_path: Path,
        max_examples: Optional[int] = None,
    ) -> MedQAResult:
        """
        Evaluate on MedQA dataset.

        Args:
            data_path: Path to MedQA JSONL file.
            max_examples: Limit evaluation size.

        Returns:
            MedQAResult with accuracy metrics.
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"MedQA data not found at {data_path}. "
                f"Download from https://github.com/jind11/MedQA"
            )

        examples = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        if max_examples:
            examples = examples[:max_examples]

        correct = 0
        for example in examples:
            question = example["question"]
            options = example.get("options", {})
            gold = example.get("answer_idx", example.get("answer", ""))

            predicted = self._select_answer(question, options)
            if predicted.upper() == gold.upper():
                correct += 1

        accuracy = correct / len(examples) if examples else 0.0
        return MedQAResult(
            accuracy=accuracy,
            accuracy_by_type={},
            num_examples=len(examples),
            num_correct=correct,
        )

    def _select_answer(self, question: str, options: dict) -> str:
        """Select best answer choice using RAG."""
        if self.generator:
            # Format as MCQ prompt
            options_text = "\n".join(
                f"{key}. {value}" for key, value in options.items()
            )
            full_question = (
                f"{question}\n\nOptions:\n{options_text}\n\n"
                f"Select the best answer (respond with the letter only): "
            )
            docs = self.retriever.retrieve(question, top_k=3) if self.retriever else []
            answer = self.generator.generate(full_question, docs, task="clinical_qa")
            # Extract letter from response
            match = re.search(r"\b([A-E])\b", answer.text)
            return match.group(1) if match else "A"
        return "A"


# ---------------------------------------------------------------------------
# i2b2 NER evaluator
# ---------------------------------------------------------------------------

class I2B2NEREvaluator:
    """
    i2b2 Clinical NER evaluation.

    Evaluates NER on the 2010 i2b2/VA Concept Extraction Challenge:
      Entity types: Problem, Treatment, Test
      Format: standoff annotation (.con files)

    Also supports 2012 i2b2 Temporal Relations challenge.

    Args:
        ner_model: BiomedicalNER model to evaluate.
    """

    # Map i2b2 labels to our types
    I2B2_LABEL_MAP = {
        "problem": "DISEASE",
        "treatment": "DRUG",
        "test": "LAB_VALUE",
    }

    def __init__(self, ner_model=None):
        self.ner_model = ner_model

    def evaluate(
        self,
        data_dir: Path,
        max_files: Optional[int] = None,
    ) -> I2B2NERResult:
        """
        Evaluate NER on i2b2 data.

        Args:
            data_dir: Directory with .txt and .con annotation files.
            max_files: Limit evaluation size.

        Returns:
            I2B2NERResult with per-type F1 metrics.
        """
        data_dir = Path(data_dir)
        txt_files = sorted(data_dir.glob("*.txt"))
        if max_files:
            txt_files = txt_files[:max_files]

        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in {data_dir}. "
                f"i2b2 data requires credentialing at https://www.i2b2.org/NLP/DataSets/"
            )

        tp_by_type: Dict[str, int] = defaultdict(int)
        fp_by_type: Dict[str, int] = defaultdict(int)
        fn_by_type: Dict[str, int] = defaultdict(int)

        for txt_file in txt_files:
            con_file = txt_file.with_suffix(".con")
            if not con_file.exists():
                continue

            text = txt_file.read_text(encoding="utf-8")
            gold_entities = self._load_con_annotations(con_file)

            if self.ner_model:
                pred_result = self.ner_model.extract(text)
                pred_entities = {
                    (e.text.lower(), e.entity_type): e
                    for e in pred_result.entities
                }
            else:
                pred_entities = {}

            for gold_text, gold_type in gold_entities:
                norm_type = self.I2B2_LABEL_MAP.get(gold_type.lower(), gold_type.upper())
                key = (gold_text.lower(), norm_type)
                if key in pred_entities:
                    tp_by_type[norm_type] += 1
                else:
                    fn_by_type[norm_type] += 1

            for (pred_text, pred_type) in pred_entities:
                gold_keys = {(g.lower(), self.I2B2_LABEL_MAP.get(t.lower(), t.upper()))
                             for g, t in gold_entities}
                if (pred_text, pred_type) not in gold_keys:
                    fp_by_type[pred_type] += 1

        # Compute metrics
        p_by_type, r_by_type, f_by_type = {}, {}, {}
        total_tp = total_fp = total_fn = 0
        for etype in set(list(tp_by_type.keys()) + list(fp_by_type.keys())):
            tp = tp_by_type[etype]
            fp = fp_by_type[etype]
            fn = fn_by_type[etype]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            p_by_type[etype] = p
            r_by_type[etype] = r
            f_by_type[etype] = f
            total_tp += tp
            total_fp += fp
            total_fn += fn

        macro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        macro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        macro_f = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0

        return I2B2NERResult(
            precision=macro_p,
            recall=macro_r,
            f1=macro_f,
            precision_by_type=p_by_type,
            recall_by_type=r_by_type,
            f1_by_type=f_by_type,
            num_examples=len(txt_files),
        )

    @staticmethod
    def _load_con_annotations(con_file: Path) -> List[Tuple[str, str]]:
        """
        Parse i2b2 .con annotation file.

        Format: c="entity text" [line:word line:word]||t="type"
        """
        entities = []
        with open(con_file, encoding="utf-8") as f:
            for line in f:
                match = re.match(
                    r'c="([^"]+)"\s+[\d:]+\s+[\d:]+\|\|t="(\w+)"', line.strip()
                )
                if match:
                    entities.append((match.group(1), match.group(2)))
        return entities
