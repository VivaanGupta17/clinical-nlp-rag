"""
Biomedical relation extraction.

Extracts structured relationships between biomedical entities:

  1. Drug-Drug Interactions (DDI):
     - Pharmacokinetic: "Fluconazole increases plasma levels of warfarin"
     - Pharmacodynamic: "NSAIDs and warfarin increase bleeding risk"
     Source: DDI Extraction 2013 corpus, DrugBank

  2. Drug-Adverse Drug Reaction (Drug-ADR):
     - "Metformin is associated with lactic acidosis"
     - "Statins may cause myopathy"
     Source: TAC 2017 ADR, SIDER, FAERS

  3. Gene-Disease Relationships:
     - "BRCA1 mutations are associated with breast cancer"
     - "TP53 is a tumor suppressor frequently mutated in lung cancer"
     Source: DisGeNET, OMIM, ClinVar

  4. Protein-Protein Interactions (PPI):
     Source: BioGRID, STRING, BioC-BioGRID

Approaches:
  - Pipeline: NER → entity pair enumeration → RE classification
  - End-to-end: Joint entity and relation extraction (DYGIE++, PL-Marker)
  - Prompt-based: LLM zero-shot/few-shot extraction for low-resource settings

References:
  - Krallinger et al. (2017). Overview of the BioCreative VI precision
    medicine track: mining protein interactions and mutations for precision
    medicine. Database.
  - Herrero-Zazo et al. (2013). The DDI corpus: An annotated corpus with
    pharmacological substances and drug-drug interactions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.ner.biomedical_ner import BiomedicalEntity, BiomedicalNER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relation data models
# ---------------------------------------------------------------------------

@dataclass
class BiologicalRelation:
    """A directed relationship between two biomedical entities."""

    subject: str          # Head entity text
    subject_type: str     # Entity type of head
    predicate: str        # Relation type
    object: str           # Tail entity text
    object_type: str      # Entity type of tail
    confidence: float
    evidence_sentence: str
    source_doc_id: Optional[str] = None
    subject_cui: Optional[str] = None
    object_cui: Optional[str] = None
    negated: bool = False  # "X does NOT interact with Y"
    speculative: bool = False  # "X may cause Y"
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        neg = "NOT " if self.negated else ""
        spec = "(speculative) " if self.speculative else ""
        return (
            f"{self.subject} [{self.subject_type}] "
            f"--{neg}{spec}{self.predicate}--> "
            f"{self.object} [{self.object_type}] "
            f"(conf={self.confidence:.2f})"
        )


@dataclass
class DrugDDIRelation(BiologicalRelation):
    """Drug-Drug Interaction relation."""

    ddi_type: str = ""  # 'effect', 'mechanism', 'advise', 'int'

    def __post_init__(self):
        self.predicate = f"DDI_{self.ddi_type}" if self.ddi_type else "DDI"
        self.subject_type = "DRUG"
        self.object_type = "DRUG"


@dataclass
class DrugADRRelation(BiologicalRelation):
    """Drug-Adverse Drug Reaction relation."""

    meddra_pt: Optional[str] = None  # MedDRA Preferred Term for ADR
    severity: Optional[str] = None  # 'serious', 'non-serious', 'unknown'
    causality: Optional[str] = None  # 'certain', 'probable', 'possible', 'unlikely'

    def __post_init__(self):
        self.predicate = "CAUSES_ADR"
        self.subject_type = "DRUG"
        self.object_type = "DISEASE"


@dataclass
class GeneDiseaseRelation(BiologicalRelation):
    """Gene-Disease association."""

    score: float = 0.0  # DisGeNET-style association score
    association_type: str = ""  # 'GeneticVariation', 'AlteredExpression', etc.

    def __post_init__(self):
        self.predicate = "ASSOCIATED_WITH"
        self.subject_type = "GENE"
        self.object_type = "DISEASE"


# ---------------------------------------------------------------------------
# Base relation extractor
# ---------------------------------------------------------------------------

class BaseRelationExtractor:
    """Abstract base for relation extractors."""

    def extract(
        self, text: str, entities: Optional[List[BiomedicalEntity]] = None
    ) -> List[BiologicalRelation]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DDI extractor
# ---------------------------------------------------------------------------

class DrugDDIExtractor(BaseRelationExtractor):
    """
    Drug-Drug Interaction (DDI) extractor.

    Two-stage pipeline:
      1. NER: identify all DRUG entities in text.
      2. Relation classification: for each drug pair, classify whether
         a DDI exists and what type.

    Uses a transformer fine-tuned on the DDI Extraction 2013 corpus:
      - 792 MedLine abstracts annotated with DDI types
      - 4 relation types: mechanism, effect, advise, int (unspecified)

    Also includes pattern-based extraction for common DDI expressions
    as a fast fallback.

    Args:
        model_name: HuggingFace DDI classification model.
        ner: BiomedicalNER instance (or None to use built-in).
        use_pattern_fallback: Use regex patterns if model unavailable.
        min_confidence: Minimum confidence for relation inclusion.
    """

    # Pattern-based DDI extraction patterns (high-precision fallback)
    _DDI_PATTERNS = [
        # Pharmacokinetic
        (r"(?P<drug1>\w+(?:\s+\w+)?)\s+(?:significantly\s+)?(?:increases?|decreases?|"
         r"reduces?|elevates?|lowers?)\s+(?:plasma\s+)?(?:levels?|concentrations?)\s+of\s+"
         r"(?P<drug2>\w+(?:\s+\w+)?)", "mechanism"),
        # Pharmacodynamic
        (r"(?P<drug1>\w+(?:\s+\w+)?)\s+(?:and|with)\s+(?P<drug2>\w+(?:\s+\w+)?)\s+"
         r"(?:may\s+)?(?:increase|decrease|cause)\s+(?:\w+\s+)?risk", "effect"),
        # Contraindication
        (r"(?:do\s+not\s+use|avoid\s+combining?|contraindicated)\s+(?P<drug1>\w+(?:\s+\w+)?)"
         r"\s+(?:with|and)\s+(?P<drug2>\w+(?:\s+\w+)?)", "advise"),
    ]

    def __init__(
        self,
        model_name: str = "DMKD/DDI-Extraction-2013",
        ner: Optional[BiomedicalNER] = None,
        use_pattern_fallback: bool = True,
        min_confidence: float = 0.6,
    ):
        self.model_name = model_name
        self.ner = ner or BiomedicalNER(use_scispacy=True, use_bert=False)
        self.use_pattern_fallback = use_pattern_fallback
        self.min_confidence = min_confidence
        self._classifier = None
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), t) for p, t in self._DDI_PATTERNS
        ]

    def _load_classifier(self):
        if self._classifier is not None:
            return
        try:
            from transformers import pipeline
            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=-1,
            )
        except Exception as exc:
            logger.warning("DDI model load failed: %s. Using pattern fallback.", exc)

    def extract(
        self,
        text: str,
        entities: Optional[List[BiomedicalEntity]] = None,
    ) -> List[DrugDDIRelation]:
        """
        Extract DDI relations from text.

        Args:
            text: Biomedical text (abstract or clinical note).
            entities: Pre-computed NER entities (optional; saves re-running NER).

        Returns:
            List of DrugDDIRelation objects.
        """
        # Get drug entities
        if entities is None:
            ner_result = self.ner.extract(text)
            entities = ner_result.entities
        drug_entities = [e for e in entities if e.entity_type == "DRUG"]

        relations: List[DrugDDIRelation] = []

        # Model-based extraction
        if len(drug_entities) >= 2:
            self._load_classifier()
            if self._classifier:
                relations.extend(
                    self._model_extract(text, drug_entities)
                )

        # Pattern-based extraction (fallback or supplement)
        if self.use_pattern_fallback:
            relations.extend(self._pattern_extract(text))

        # Deduplicate
        relations = self._deduplicate_relations(relations)
        return [r for r in relations if r.confidence >= self.min_confidence]

    def _model_extract(
        self, text: str, drug_entities: List[BiomedicalEntity]
    ) -> List[DrugDDIRelation]:
        """Enumerate drug pairs and classify with RE model."""
        relations = []
        sentences = self._split_sentences(text)

        for sent in sentences:
            # Find drug mentions in this sentence
            sent_drugs = [
                e for e in drug_entities
                if e.text.lower() in sent.lower()
            ]
            # Enumerate pairs
            for i, drug1 in enumerate(sent_drugs):
                for drug2 in sent_drugs[i + 1:]:
                    # Create marked sentence: replace drug mentions with [DRUG1]/[DRUG2]
                    marked = sent.replace(drug1.text, "[DRUG1]", 1)
                    marked = marked.replace(drug2.text, "[DRUG2]", 1)

                    try:
                        result = self._classifier(marked, truncation=True, max_length=256)
                        label = result[0]["label"].upper()
                        score = float(result[0]["score"])

                        if label not in ("NONE", "O", "FALSE"):
                            relations.append(
                                DrugDDIRelation(
                                    subject=drug1.text,
                                    subject_type="DRUG",
                                    predicate=f"DDI_{label}",
                                    object=drug2.text,
                                    object_type="DRUG",
                                    confidence=score,
                                    evidence_sentence=sent,
                                    ddi_type=label.lower(),
                                    model=self.model_name,
                                )
                            )
                    except Exception as exc:
                        logger.debug("DDI classification error: %s", exc)

        return relations

    def _pattern_extract(self, text: str) -> List[DrugDDIRelation]:
        """Extract DDIs using regular expression patterns."""
        relations = []
        for pattern, ddi_type in self._compiled_patterns:
            for match in pattern.finditer(text):
                drug1 = match.group("drug1").strip()
                drug2 = match.group("drug2").strip()
                # Get surrounding sentence as evidence
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                relations.append(
                    DrugDDIRelation(
                        subject=drug1,
                        subject_type="DRUG",
                        predicate=f"DDI_{ddi_type}",
                        object=drug2,
                        object_type="DRUG",
                        confidence=0.70,
                        evidence_sentence=text[start:end],
                        ddi_type=ddi_type,
                        model="pattern",
                    )
                )
        return relations

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @staticmethod
    def _deduplicate_relations(
        relations: List[DrugDDIRelation],
    ) -> List[DrugDDIRelation]:
        """Remove duplicate drug pairs, keeping highest confidence."""
        seen: Dict[Tuple, DrugDDIRelation] = {}
        for rel in relations:
            key = (rel.subject.lower(), rel.object.lower(), rel.ddi_type)
            if key not in seen or rel.confidence > seen[key].confidence:
                seen[key] = rel
        return list(seen.values())


# ---------------------------------------------------------------------------
# Drug-ADR extractor
# ---------------------------------------------------------------------------

class DrugADRExtractor(BaseRelationExtractor):
    """
    Drug-Adverse Drug Reaction (ADR) extractor.

    Extracts drug-adverse event associations from:
      - Clinical notes (post-market surveillance)
      - PubMed case reports and observational studies
      - FDA FAERS narratives

    Patterns capture:
      - "Drug X caused/led to/resulted in ADR Y"
      - "Patient experienced Y after taking X"
      - "Y was reported in patients receiving X"

    Args:
        ner: BiomedicalNER for entity extraction.
        min_confidence: Minimum confidence threshold.
        extract_negated: Include negated relations (e.g., "no adverse events").
        normalize_adr: Normalize ADR mentions to MedDRA Preferred Terms.
    """

    _ADR_PATTERNS = [
        (r"(?P<drug>\w+(?:\s+\w+)?)\s+(?:may\s+)?(?:cause[sd]?|induce[sd]?|"
         r"result(?:ed)?\s+in|lead[s]?\s+to|associated\s+with)\s+(?P<adr>\w+(?:\s+\w+)?)", 0.75),
        (r"(?P<adr>\w+(?:\s+\w+)?)\s+(?:was|were|has\s+been)\s+(?:reported|"
         r"observed|noted)\s+(?:in\s+patients\s+)?(?:receiving|taking|treated\s+with)\s+"
         r"(?P<drug>\w+(?:\s+\w+)?)", 0.70),
        (r"side\s+effects?\s+(?:of|associated\s+with)\s+(?P<drug>\w+(?:\s+\w+)?):\s+"
         r"(?P<adr>[^.;]+)", 0.65),
    ]

    def __init__(
        self,
        ner: Optional[BiomedicalNER] = None,
        min_confidence: float = 0.6,
        extract_negated: bool = False,
        normalize_adr: bool = False,
    ):
        self.ner = ner or BiomedicalNER(use_scispacy=True)
        self.min_confidence = min_confidence
        self.extract_negated = extract_negated
        self.normalize_adr = normalize_adr
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), conf)
            for p, conf in self._ADR_PATTERNS
        ]

    def extract(
        self,
        text: str,
        entities: Optional[List[BiomedicalEntity]] = None,
    ) -> List[DrugADRRelation]:
        """
        Extract drug-ADR relations from text.

        Args:
            text: Clinical or biomedical text.
            entities: Pre-computed entities (optional).

        Returns:
            List of DrugADRRelation objects.
        """
        if entities is None:
            ner_result = self.ner.extract(text)
            entities = ner_result.entities

        relations: List[DrugADRRelation] = []

        # Pattern-based extraction
        for pattern, conf in self._compiled_patterns:
            for match in pattern.finditer(text):
                try:
                    drug = match.group("drug").strip()
                    adr = match.group("adr").strip()

                    # Check for negation context
                    start = max(0, match.start() - 30)
                    context = text[start : match.start()]
                    negated = bool(
                        re.search(
                            r"\b(no|not|without|negative for|denies?)\b",
                            context,
                            re.IGNORECASE,
                        )
                    )
                    if negated and not self.extract_negated:
                        continue

                    # Check for speculative language
                    speculative = bool(
                        re.search(
                            r"\b(may|might|could|possibly|potentially|suspected)\b",
                            text[match.start(): match.end()],
                            re.IGNORECASE,
                        )
                    )

                    # Evidence sentence
                    sent_start = text.rfind(".", 0, match.start()) + 1
                    sent_end = text.find(".", match.end()) + 1
                    evidence = text[sent_start:sent_end].strip() if sent_end > 0 else text[match.start():match.end()]

                    relations.append(
                        DrugADRRelation(
                            subject=drug,
                            subject_type="DRUG",
                            predicate="CAUSES_ADR",
                            object=adr,
                            object_type="DISEASE",
                            confidence=conf * (0.8 if speculative else 1.0),
                            evidence_sentence=evidence,
                            negated=negated,
                            speculative=speculative,
                            model="pattern",
                        )
                    )
                except IndexError:
                    pass

        return [r for r in relations if r.confidence >= self.min_confidence]


# ---------------------------------------------------------------------------
# Gene-Disease extractor
# ---------------------------------------------------------------------------

class GeneDiseaseExtractor(BaseRelationExtractor):
    """
    Gene-Disease relationship extractor.

    Extracts associations between genes/proteins and diseases, including:
      - Causal: "BRCA1 mutations cause hereditary breast cancer"
      - Associative: "TP53 is frequently altered in colorectal cancer"
      - Regulatory: "VEGF promotes tumor angiogenesis in glioblastoma"

    Uses a BERT model fine-tuned on the BioRED dataset (gene-disease,
    chemical-disease, gene-chemical relations) or DisGeNET text corpus.

    Args:
        ner: BiomedicalNER instance.
        model_name: HuggingFace RE model.
        min_confidence: Minimum relation confidence.
    """

    _GD_PATTERNS = [
        (r"(?P<gene>[A-Z][A-Z0-9]+(?:\d+)?)\s+(?:mutations?\s+)?(?:are|is)\s+"
         r"(?:associated\s+with|linked\s+to|implicated\s+in)\s+(?P<disease>\w+(?:\s+\w+){1,4})", 0.75),
        (r"(?P<gene>[A-Z][A-Z0-9]+(?:\d+)?)\s+(?:gene\s+)?(?:mutations?|variants?|"
         r"alterations?)\s+(?:cause|result\s+in|lead\s+to)\s+(?P<disease>\w+(?:\s+\w+){1,4})", 0.80),
        (r"(?P<disease>\w+(?:\s+\w+){1,4})\s+(?:is|are)\s+caused\s+by\s+"
         r"(?:mutations?\s+in\s+)?(?P<gene>[A-Z][A-Z0-9]+(?:\d+)?)", 0.80),
    ]

    def __init__(
        self,
        ner: Optional[BiomedicalNER] = None,
        min_confidence: float = 0.6,
    ):
        self.ner = ner or BiomedicalNER(use_scispacy=True)
        self.min_confidence = min_confidence
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), conf)
            for p, conf in self._GD_PATTERNS
        ]

    def extract(
        self,
        text: str,
        entities: Optional[List[BiomedicalEntity]] = None,
    ) -> List[GeneDiseaseRelation]:
        """Extract gene-disease relations."""
        relations: List[GeneDiseaseRelation] = []

        for pattern, conf in self._compiled_patterns:
            for match in pattern.finditer(text):
                try:
                    gene = match.group("gene").strip()
                    disease = match.group("disease").strip()
                    relations.append(
                        GeneDiseaseRelation(
                            subject=gene,
                            subject_type="GENE",
                            predicate="ASSOCIATED_WITH",
                            object=disease,
                            object_type="DISEASE",
                            confidence=conf,
                            evidence_sentence=text[
                                max(0, match.start() - 20): match.end() + 20
                            ].strip(),
                            model="pattern",
                        )
                    )
                except IndexError:
                    pass

        return [r for r in relations if r.confidence >= self.min_confidence]


# ---------------------------------------------------------------------------
# Unified relation extractor
# ---------------------------------------------------------------------------

class BiomedicalRelationExtractor:
    """
    Unified relation extractor combining DDI, Drug-ADR, and Gene-Disease.

    Args:
        extract_ddi: Include drug-drug interactions.
        extract_adr: Include drug-adverse reactions.
        extract_gene_disease: Include gene-disease associations.
        ner: Shared BiomedicalNER instance.
        min_confidence: Global minimum confidence threshold.
    """

    def __init__(
        self,
        extract_ddi: bool = True,
        extract_adr: bool = True,
        extract_gene_disease: bool = True,
        ner: Optional[BiomedicalNER] = None,
        min_confidence: float = 0.6,
    ):
        shared_ner = ner or BiomedicalNER(use_scispacy=True)
        self._extractors = []
        if extract_ddi:
            self._extractors.append(DrugDDIExtractor(ner=shared_ner, min_confidence=min_confidence))
        if extract_adr:
            self._extractors.append(DrugADRExtractor(ner=shared_ner, min_confidence=min_confidence))
        if extract_gene_disease:
            self._extractors.append(GeneDiseaseExtractor(ner=shared_ner, min_confidence=min_confidence))

    def extract_all(self, text: str) -> Dict[str, List[BiologicalRelation]]:
        """
        Extract all configured relation types.

        Args:
            text: Text to process.

        Returns:
            Dict mapping relation type to list of relations.
        """
        # Run NER once and share across extractors
        ner_result = None
        try:
            ner_result = self._extractors[0].ner.extract(text) if self._extractors else None
        except Exception:
            pass

        entities = ner_result.entities if ner_result else None

        results: Dict[str, List[BiologicalRelation]] = {}
        for extractor in self._extractors:
            try:
                relations = extractor.extract(text, entities=entities)
                rel_type = type(extractor).__name__.replace("Extractor", "")
                results[rel_type] = relations
            except Exception as exc:
                logger.warning(
                    "Extractor %s failed: %s", type(extractor).__name__, exc
                )

        return results

    def extract_structured(self, text: str) -> dict:
        """
        Extract and return relations in structured JSON-compatible format.

        Returns:
            Dict with 'ddi', 'drug_adr', 'gene_disease' keys, each containing
            a list of relation dicts.
        """
        raw = self.extract_all(text)
        return {
            key: [r.to_dict() for r in rels]
            for key, rels in raw.items()
        }
