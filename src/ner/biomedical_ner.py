"""
Biomedical Named Entity Recognition (NER).

Extracts structured biomedical entities from clinical text and literature:
  - DISEASE — disorders, syndromes, conditions (e.g., "myocardial infarction")
  - DRUG — medications, chemical compounds (e.g., "atorvastatin", "metformin")
  - GENE — gene and protein names (e.g., "BRCA1", "TP53")
  - PROTEIN — protein names and domains
  - MUTATION — genetic variants (e.g., "V600E", "c.1799T>A")
  - CELL_LINE — cell line names (e.g., "HeLa", "MCF-7")
  - SPECIES — organism names (e.g., "Homo sapiens", "E. coli")
  - CHEMICAL — chemical compounds (non-drug)
  - PROCEDURE — clinical procedures and interventions

Implementation stack:
  1. ScispaCy — spaCy models fine-tuned on biomedical text (Allen AI).
     Fast, production-ready. Uses en_core_sci_lg or en_ner_bc5cdr_md.
  2. HuggingFace BERT-based NER — fine-tuned on BC5CDR (drugs/diseases),
     NCBI Disease, and i2b2 datasets. More accurate, slower.
  3. UMLS Entity Linking — maps extracted entities to UMLS CUIs using
     scispaCy's EntityLinker or QuickUMLS.

Training data:
  - BC5CDR: 1,500 PubMed abstracts, drug and disease entities
  - NCBI Disease: 793 PubMed abstracts, disease mentions
  - i2b2 2010: Clinical text, problem/treatment/test entities
  - LivingNER: Species mentions in clinical notes

References:
  - Neumann et al. (2019). ScispaCy: Fast and Robust Models for Biomedical NLP.
    ACL BioNLP Workshop.
  - Li, J. et al. (2016). BioCreative V CDR Task Corpus. Database.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Entity type constants
ENTITY_TYPES = {
    "DISEASE": "Disease or syndrome",
    "DRUG": "Drug or pharmacological substance",
    "GENE": "Gene name",
    "PROTEIN": "Protein name",
    "MUTATION": "Genetic variant or mutation",
    "CELL_LINE": "Cell line name",
    "SPECIES": "Organism or species",
    "CHEMICAL": "Chemical compound",
    "PROCEDURE": "Clinical procedure or intervention",
    "ANATOMY": "Anatomical structure",
    "LAB_VALUE": "Laboratory test or measurement",
}


# ---------------------------------------------------------------------------
# Entity data model
# ---------------------------------------------------------------------------

@dataclass
class BiomedicalEntity:
    """
    A recognized biomedical named entity.

    Attributes:
        text: Surface form as it appears in the text.
        entity_type: Entity type (DISEASE, DRUG, GENE, etc.).
        start_char: Character offset of entity start.
        end_char: Character offset of entity end.
        confidence: Model confidence (0–1).
        normalized_form: Normalized/canonical form of the entity.
        umls_cui: UMLS Concept Unique Identifier.
        umls_name: Canonical UMLS concept name.
        umls_semantic_type: UMLS semantic type (e.g., "T047" = Disease).
        source_model: Which model produced this entity.
    """

    text: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    normalized_form: Optional[str] = None
    umls_cui: Optional[str] = None
    umls_name: Optional[str] = None
    umls_semantic_type: Optional[str] = None
    source_model: str = "scispacy"

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        parts = [f"{self.text} [{self.entity_type}]"]
        if self.umls_cui:
            parts.append(f"CUI:{self.umls_cui}")
        if self.umls_name and self.umls_name != self.text:
            parts.append(f"→ {self.umls_name}")
        return " ".join(parts)


@dataclass
class NERResult:
    """Result from NER over a single document."""

    text: str
    entities: List[BiomedicalEntity] = field(default_factory=list)
    entities_by_type: Dict[str, List[BiomedicalEntity]] = field(default_factory=dict)
    model_used: str = ""

    def __post_init__(self):
        if self.entities and not self.entities_by_type:
            self.entities_by_type = self._group_by_type()

    def _group_by_type(self) -> Dict[str, List[BiomedicalEntity]]:
        grouped: Dict[str, List[BiomedicalEntity]] = {}
        for ent in self.entities:
            grouped.setdefault(ent.entity_type, []).append(ent)
        return grouped

    @property
    def drugs(self) -> List[BiomedicalEntity]:
        return self.entities_by_type.get("DRUG", [])

    @property
    def diseases(self) -> List[BiomedicalEntity]:
        return self.entities_by_type.get("DISEASE", [])

    @property
    def genes(self) -> List[BiomedicalEntity]:
        return self.entities_by_type.get("GENE", [])

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "counts": {k: len(v) for k, v in self.entities_by_type.items()},
            "model_used": self.model_used,
        }


# ---------------------------------------------------------------------------
# ScispaCy NER
# ---------------------------------------------------------------------------

class ScispaCyNER:
    """
    Biomedical NER using ScispaCy models.

    ScispaCy provides spaCy-compatible models trained on biomedical text:
      - en_core_sci_sm: Small (100MB), good for general biomedical text
      - en_core_sci_md: Medium, includes word vectors
      - en_core_sci_lg: Large, best accuracy
      - en_ner_bc5cdr_md: BC5CDR-specific (drugs + diseases)
      - en_ner_craft_md: CRAFT corpus (genes, proteins, chemicals)
      - en_ner_jnlpba_md: JNLPBA corpus (proteins, DNA, RNA, cell lines)
      - en_ner_bionlp13cg_md: BioNLP 2013 (cancer genetics)

    EntityLinker:
      Optionally adds UMLS entity linking to map surface forms to CUIs.
      Requires: pip install scispacy && python -m spacy download en_core_sci_lg
      UMLS linker: pip install https://s3.../en_core_sci_lg.tar.gz

    Args:
        model_name: ScispaCy model name.
        add_entity_linker: Link entities to UMLS concepts.
        linker_name: 'umls', 'mesh', 'rxnorm', 'go', or 'hpo'.
        resolve_abbreviations: Use scispaCy AbbreviationDetector.
        entity_type_map: Map scispaCy labels to our standard types.
    """

    # Map scispaCy entity labels to our standard types
    _LABEL_MAP = {
        "DISEASE": "DISEASE",
        "CHEMICAL": "DRUG",
        "DRUG": "DRUG",
        "GENE_OR_GENE_PRODUCT": "GENE",
        "PROTEIN": "PROTEIN",
        "DNA": "GENE",
        "RNA": "GENE",
        "CELL_LINE": "CELL_LINE",
        "CELL_TYPE": "ANATOMY",
        "ORGANISM": "SPECIES",
        "MUTATION": "MUTATION",
        # BioNLP13CG labels
        "GENE": "GENE",
        "AMINO_ACID": "PROTEIN",
        "CANCER": "DISEASE",
        "ORGAN": "ANATOMY",
        "TISSUE": "ANATOMY",
        # Generic
        "ENTITY": "DISEASE",  # en_core_sci models use ENTITY
    }

    def __init__(
        self,
        model_name: str = "en_core_sci_lg",
        add_entity_linker: bool = False,
        linker_name: str = "umls",
        resolve_abbreviations: bool = True,
    ):
        self.model_name = model_name
        self.add_entity_linker = add_entity_linker
        self.linker_name = linker_name
        self.resolve_abbreviations = resolve_abbreviations
        self._nlp = None

    def _load_model(self):
        if self._nlp is not None:
            return
        try:
            import spacy
            self._nlp = spacy.load(self.model_name)

            if self.resolve_abbreviations:
                try:
                    from scispacy.abbreviation import AbbreviationDetector
                    self._nlp.add_pipe("abbreviation_detector")
                except Exception:
                    pass

            if self.add_entity_linker:
                try:
                    self._nlp.add_pipe(
                        "scispacy_linker",
                        config={
                            "resolve_abbreviations": True,
                            "linker_name": self.linker_name,
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to add entity linker '%s': %s. "
                        "Install scispaCy UMLS KB.",
                        self.linker_name,
                        exc,
                    )
            logger.info("ScispaCy model loaded: %s", self.model_name)
        except OSError as exc:
            raise OSError(
                f"ScispaCy model '{self.model_name}' not found.\n"
                f"Install with: pip install https://s3-us-west-2.amazonaws.com/"
                f"ai2-s2-scispacy/releases/v0.5.3/{self.model_name}-0.5.3.tar.gz"
            ) from exc

    def extract(self, text: str) -> NERResult:
        """
        Extract biomedical entities from text.

        Args:
            text: Clinical or biomedical text to process.

        Returns:
            NERResult with recognized entities and UMLS links.
        """
        self._load_model()
        doc = self._nlp(text)

        entities: List[BiomedicalEntity] = []
        for ent in doc.ents:
            entity_type = self._LABEL_MAP.get(ent.label_, ent.label_)

            # UMLS linking
            umls_cui = None
            umls_name = None
            umls_sem_type = None
            if self.add_entity_linker and hasattr(ent._, "kb_ents") and ent._.kb_ents:
                best = ent._.kb_ents[0]  # Highest-scoring UMLS match
                umls_cui = best[0]
                umls_score = best[1]
                try:
                    linker = self._nlp.get_pipe("scispacy_linker")
                    kb_entity = linker.kb.cui_to_entity.get(umls_cui)
                    if kb_entity:
                        umls_name = kb_entity.canonical_name
                        umls_sem_type = (
                            kb_entity.types[0] if kb_entity.types else None
                        )
                except Exception:
                    pass

            entities.append(
                BiomedicalEntity(
                    text=ent.text,
                    entity_type=entity_type,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=1.0,  # spaCy doesn't expose per-entity confidence
                    umls_cui=umls_cui,
                    umls_name=umls_name,
                    umls_semantic_type=umls_sem_type,
                    source_model=self.model_name,
                )
            )

        return NERResult(text=text, entities=entities, model_used=self.model_name)

    def extract_batch(self, texts: List[str]) -> List[NERResult]:
        """Batch entity extraction using spaCy pipe (efficient)."""
        self._load_model()
        results = []
        for doc in self._nlp.pipe(texts, batch_size=32):
            entities = []
            for ent in doc.ents:
                entity_type = self._LABEL_MAP.get(ent.label_, ent.label_)
                entities.append(
                    BiomedicalEntity(
                        text=ent.text,
                        entity_type=entity_type,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        source_model=self.model_name,
                    )
                )
            results.append(
                NERResult(text=doc.text, entities=entities, model_used=self.model_name)
            )
        return results


# ---------------------------------------------------------------------------
# HuggingFace BERT-based NER
# ---------------------------------------------------------------------------

class BERTBiomedicalNER:
    """
    BERT-based biomedical NER using HuggingFace models.

    Fine-tuned models for each entity type:
      - Drug + Disease: alvaroalon2/biobert_diseases_ner (BC5CDR-trained)
      - Gene: fran-martinez/scibert-finetuned-ner (CRAFT)
      - Clinical: samrawal/bert-base-uncased_clinical-ner (i2b2)
      - General biomedical: d4data/biomedical-ner-all

    Args:
        model_name: HuggingFace model name or path.
        device: Inference device.
        batch_size: Inference batch size.
        aggregation_strategy: 'simple' or 'first' or 'average' or 'max'.
    """

    def __init__(
        self,
        model_name: str = "d4data/biomedical-ner-all",
        device: str = "auto",
        batch_size: int = 8,
        aggregation_strategy: str = "simple",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.aggregation_strategy = aggregation_strategy
        self._pipeline = None

        if device == "auto":
            try:
                import torch
                self._device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                self._device = -1
        else:
            self._device = 0 if device == "cuda" else -1

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy=self.aggregation_strategy,
                device=self._device,
            )
            logger.info("BERT NER pipeline loaded: %s", self.model_name)
        except ImportError as exc:
            raise ImportError(
                "transformers required. Install: pip install transformers"
            ) from exc

    def extract(self, text: str) -> NERResult:
        """Extract entities using BERT NER pipeline."""
        self._load_pipeline()
        outputs = self._pipeline(text)

        entities = []
        for ent in outputs:
            entity_type = ent["entity_group"].upper()
            entities.append(
                BiomedicalEntity(
                    text=ent["word"],
                    entity_type=entity_type,
                    start_char=ent["start"],
                    end_char=ent["end"],
                    confidence=float(ent["score"]),
                    source_model=self.model_name,
                )
            )

        return NERResult(text=text, entities=entities, model_used=self.model_name)


# ---------------------------------------------------------------------------
# Ensemble NER
# ---------------------------------------------------------------------------

class BiomedicalNER:
    """
    Ensemble biomedical NER combining ScispaCy and BERT models.

    Combines predictions from multiple models, resolving conflicts by taking
    the highest-confidence prediction for overlapping spans.

    Args:
        use_scispacy: Include ScispaCy model.
        use_bert: Include BERT NER model.
        scispacy_model: ScispaCy model name.
        bert_model: BERT NER model name.
        add_umls_linking: Link entities to UMLS concepts.
        min_confidence: Minimum confidence for entity inclusion.
        deduplicate: Remove overlapping spans (keep highest confidence).

    Example::

        ner = BiomedicalNER(add_umls_linking=True)
        result = ner.extract(clinical_note_text)
        print(f"Drugs: {[e.text for e in result.drugs]}")
        print(f"Diseases: {[e.text for e in result.diseases]}")
    """

    def __init__(
        self,
        use_scispacy: bool = True,
        use_bert: bool = False,
        scispacy_model: str = "en_ner_bc5cdr_md",
        bert_model: str = "d4data/biomedical-ner-all",
        add_umls_linking: bool = False,
        min_confidence: float = 0.5,
        deduplicate: bool = True,
    ):
        self.min_confidence = min_confidence
        self.deduplicate = deduplicate
        self._models = []

        if use_scispacy:
            self._models.append(
                ScispaCyNER(
                    model_name=scispacy_model,
                    add_entity_linker=add_umls_linking,
                )
            )
        if use_bert:
            self._models.append(BERTBiomedicalNER(model_name=bert_model))

    def extract(self, text: str) -> NERResult:
        """
        Extract entities using all configured models.

        Args:
            text: Text to process.

        Returns:
            NERResult with merged entities from all models.
        """
        if not self._models:
            raise ValueError("No NER models configured.")

        all_entities: List[BiomedicalEntity] = []
        for model in self._models:
            try:
                result = model.extract(text)
                all_entities.extend(result.entities)
            except Exception as exc:
                logger.warning("NER model %s failed: %s", type(model).__name__, exc)

        # Filter by confidence
        all_entities = [
            e for e in all_entities if e.confidence >= self.min_confidence
        ]

        # Deduplicate overlapping spans
        if self.deduplicate:
            all_entities = self._deduplicate(all_entities)

        return NERResult(
            text=text,
            entities=all_entities,
            model_used="+".join(type(m).__name__ for m in self._models),
        )

    @staticmethod
    def _deduplicate(entities: List[BiomedicalEntity]) -> List[BiomedicalEntity]:
        """
        Remove overlapping entity spans by keeping the highest-confidence entity.

        Uses a greedy interval scheduling approach: sort by confidence (desc),
        add each entity if it doesn't overlap with already-selected entities.
        """
        sorted_ents = sorted(entities, key=lambda e: -e.confidence)
        selected: List[BiomedicalEntity] = []
        for ent in sorted_ents:
            overlap = any(
                e.start_char < ent.end_char and ent.start_char < e.end_char
                for e in selected
            )
            if not overlap:
                selected.append(ent)
        # Re-sort by position
        selected.sort(key=lambda e: e.start_char)
        return selected

    def extract_structured(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities and return as a structured dict.

        Returns:
            Dict mapping entity type → list of entity text strings.
        """
        result = self.extract(text)
        structured: Dict[str, List[str]] = {}
        for ent in result.entities:
            structured.setdefault(ent.entity_type, [])
            if ent.text not in structured[ent.entity_type]:
                structured[ent.entity_type].append(ent.text)
        return structured


# ---------------------------------------------------------------------------
# NER model fine-tuning
# ---------------------------------------------------------------------------

class BioNERFineTuner:
    """
    Fine-tune a BERT-based NER model on custom biomedical annotations.

    Uses HuggingFace Trainer for token classification. Input data should
    follow the BIO (Beginning-Inside-Outside) tagging scheme.

    Supported base models:
      - microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract (best quality)
      - dmis-lab/biobert-v1.1 (widely used baseline)
      - allenai/scibert_scivocab_uncased (scientific text)

    Args:
        base_model: Pre-trained model to fine-tune.
        output_dir: Directory to save fine-tuned model.
        label_list: NER label list in BIO format (e.g., ['O', 'B-DRUG', 'I-DRUG']).
        learning_rate: Training learning rate.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per device.
    """

    def __init__(
        self,
        base_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        output_dir: str = "models/ner_finetuned",
        label_list: Optional[List[str]] = None,
        learning_rate: float = 2e-5,
        num_train_epochs: int = 5,
        per_device_train_batch_size: int = 16,
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.label_list = label_list or [
            "O",
            "B-DISEASE", "I-DISEASE",
            "B-DRUG", "I-DRUG",
            "B-GENE", "I-GENE",
            "B-PROTEIN", "I-PROTEIN",
            "B-MUTATION", "I-MUTATION",
        ]
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size

        self._id2label = {i: l for i, l in enumerate(self.label_list)}
        self._label2id = {l: i for i, l in enumerate(self.label_list)}

    def train(
        self,
        train_dataset,
        eval_dataset=None,
    ) -> None:
        """
        Fine-tune NER model.

        Args:
            train_dataset: HuggingFace Dataset with 'tokens' and 'ner_tags' columns.
            eval_dataset: Optional evaluation dataset.
        """
        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                DataCollatorForTokenClassification,
                Trainer,
                TrainingArguments,
            )
            import evaluate
        except ImportError as exc:
            raise ImportError(
                "transformers and evaluate packages required. "
                "Install: pip install transformers evaluate seqeval"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForTokenClassification.from_pretrained(
            self.base_model,
            num_labels=len(self.label_list),
            id2label=self._id2label,
            label2id=self._label2id,
        )

        def tokenize_and_align(examples):
            tokenized = tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True,
            )
            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized.word_ids(batch_index=i)
                prev_word_id = None
                label_ids = []
                for word_id in word_ids:
                    if word_id is None:
                        label_ids.append(-100)
                    elif word_id != prev_word_id:
                        label_ids.append(label[word_id])
                    else:
                        # Subword: copy label but convert B → I
                        lbl = label[word_id]
                        if self.label_list[lbl].startswith("B-"):
                            lbl = self._label2id.get(
                                "I-" + self.label_list[lbl][2:], lbl
                            )
                        label_ids.append(lbl)
                    prev_word_id = word_id
                labels.append(label_ids)
            tokenized["labels"] = labels
            return tokenized

        tokenized_train = train_dataset.map(tokenize_and_align, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_and_align, batched=True) if eval_dataset else None

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=bool(eval_dataset),
            metric_for_best_model="f1",
            logging_steps=100,
            report_to="none",  # Disable wandb for clean runs
        )

        seqeval = evaluate.load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            import numpy as np
            predictions = np.argmax(predictions, axis=2)
            true_labels = [
                [self.label_list[l] for l in label if l != -100]
                for label in labels
            ]
            true_preds = [
                [
                    self.label_list[p]
                    for p, l in zip(prediction, label)
                    if l != -100
                ]
                for prediction, label in zip(predictions, labels)
            ]
            results = seqeval.compute(predictions=true_preds, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=compute_metrics,
        )

        logger.info(
            "Starting NER fine-tuning: %s → %s",
            self.base_model,
            self.output_dir,
        )
        trainer.train()
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        logger.info("NER model saved to %s", self.output_dir)
