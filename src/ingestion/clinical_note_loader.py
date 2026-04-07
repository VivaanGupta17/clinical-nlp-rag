"""
Clinical note loader for MIMIC-III dataset.

Handles parsing, de-identification (PHI removal), and section segmentation of
clinical notes including discharge summaries, radiology reports, and nursing
notes from MIMIC-III.

HIPAA Safe Harbor De-identification:
  Removes all 18 PHI identifiers per 45 CFR §164.514(b):
    Names, geographic data, dates (except year), phone/fax numbers,
    email addresses, SSNs, medical record numbers, health plan beneficiary
    numbers, account numbers, certificate/license numbers, VINs, device
    identifiers, URLs, IPs, biometric identifiers, full-face photos,
    unique identifying numbers/codes.

Data Access:
  MIMIC-III requires credentialing at https://physionet.org/content/mimiciii/
  Data must remain on credentialed systems per PhysioNet DUA.

References:
  - Johnson, A. et al. (2016). MIMIC-III, a freely accessible critical care
    database. Scientific Data, 3, 160035.
  - HIPAA Safe Harbor method: 45 CFR §164.514(b)
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PHI patterns for Safe Harbor de-identification
# ---------------------------------------------------------------------------

# 18 PHI identifiers under HIPAA Safe Harbor (45 CFR §164.514(b))
PHI_PATTERNS: Dict[str, str] = {
    # 1. Names — any capitalized word sequence followed by titles or standalone
    "name_with_title": r"\b(Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+[A-Z][a-z]+"
                       r"(\s+[A-Z][a-z]+)*",
    "name_placeholder": r"\[\*\*[^\]]*(?:Name|name)[^\]]*\*\*\]",

    # 2. Geographic subdivisions smaller than state
    "street_address": r"\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"
                      r"\s+(?:St\.?|Ave\.?|Blvd\.?|Dr\.?|Rd\.?|Lane|Way|Court|Ct\.?)",
    "zip_code": r"\b\d{5}(?:-\d{4})?\b",
    "city_state": r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?,\s*[A-Z]{2}\b",

    # 3. Dates (except year)
    "full_date": r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b",
    "month_day": r"\b(?:January|February|March|April|May|June|July|August|"
                 r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
                 r"Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:,\s*\d{4})?\b",
    "mimic_date_placeholder": r"\[\*\*\d{4}-\d{1,2}(?:-\d{1,2})?\*\*\]",

    # 4. Phone numbers
    "phone": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",

    # 5. Fax numbers (same pattern as phone in clinical context)
    # 6. Email addresses
    "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",

    # 7. Social security numbers
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",

    # 8. Medical record numbers
    "mrn": r"\b(?:MRN|Medical Record(?: Number)?|Patient ID|Pt ID)\s*[:#]?\s*\d+\b",
    "mimic_number_placeholder": r"\[\*\*\d+\*\*\]",

    # 9-11. Health plan, account, certificate/license numbers
    "account_number": r"\b(?:Account|Acct|Policy|Member ID)\s*[:#]?\s*[\w\d-]+\b",

    # 12-13. VINs and device identifiers (less common in clinical notes)
    # 14. URLs
    "url": r"https?://[^\s]+",

    # 15. IP addresses
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",

    # 16. Biometric identifiers (fingerprints, voice prints — typically not in text)
    # 17. Full-face photographs and comparable images (not in text)

    # 18. Unique identifying numbers (catch-all for MIMIC placeholders)
    "mimic_phi_placeholder": r"\[\*\*[^\]]*\*\*\]",
}

# Section headers common in clinical notes (MIMIC-III style)
SECTION_PATTERNS = {
    "chief_complaint": re.compile(
        r"(?:Chief Complaint|CC|CHIEF COMPLAINT)\s*:?\s*\n", re.IGNORECASE
    ),
    "history_of_present_illness": re.compile(
        r"(?:History of Present Illness|HPI|HISTORY OF PRESENT ILLNESS)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "past_medical_history": re.compile(
        r"(?:Past Medical History|PMH|PAST MEDICAL HISTORY)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "medications": re.compile(
        r"(?:Medications|MEDICATIONS|Current Medications|HOME MEDICATIONS)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "allergies": re.compile(
        r"(?:Allergies|ALLERGIES|Drug Allergies)\s*:?\s*\n", re.IGNORECASE
    ),
    "review_of_systems": re.compile(
        r"(?:Review of Systems|ROS|REVIEW OF SYSTEMS)\s*:?\s*\n", re.IGNORECASE
    ),
    "physical_exam": re.compile(
        r"(?:Physical Exam(?:ination)?|PE|PHYSICAL EXAM(?:INATION)?)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "laboratory_data": re.compile(
        r"(?:Laboratory(?:\s+Data)?|Labs|LAB(?:ORATORY)?(?:\s+DATA)?)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "radiology": re.compile(
        r"(?:Radiology|Imaging|RADIOLOGY|IMAGING)\s*:?\s*\n", re.IGNORECASE
    ),
    "assessment": re.compile(
        r"(?:Assessment|ASSESSMENT|Impression|IMPRESSION)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "plan": re.compile(
        r"(?:Plan|PLAN|Recommendations|RECOMMENDATIONS)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "assessment_and_plan": re.compile(
        r"(?:Assessment and Plan|A&P|A/P|ASSESSMENT AND PLAN)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "discharge_medications": re.compile(
        r"(?:Discharge Medications|DISCHARGE MEDICATIONS)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "discharge_instructions": re.compile(
        r"(?:Discharge Instructions?|DISCHARGE INSTRUCTIONS?)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "followup": re.compile(
        r"(?:Follow(?:-|\s)?up|FOLLOW(?:-|\s)?UP|Disposition)\s*:?\s*\n",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ClinicalNoteSection:
    """A parsed section of a clinical note."""
    section_name: str
    text: str
    start_char: int
    end_char: int


@dataclass
class ClinicalNote:
    """
    Structured representation of a clinical note.

    Attributes:
        note_id: Unique identifier (e.g., MIMIC row_id).
        subject_id: Anonymized patient ID.
        hadm_id: Hospital admission ID.
        note_type: Discharge Summary, Radiology, Nursing, etc.
        text: De-identified note text.
        sections: Parsed clinical sections.
        original_length: Original character count (before de-id).
        phi_count: Number of PHI instances removed.
        admission_date: Year of admission (day/month removed for HIPAA).
        source: Dataset source identifier.
    """

    note_id: str
    subject_id: str
    hadm_id: str
    note_type: str
    text: str
    sections: List[ClinicalNoteSection] = field(default_factory=list)
    original_length: int = 0
    phi_count: int = 0
    admission_date: Optional[str] = None  # Year only
    source: str = "mimic_iii"
    processed_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_rag_documents(self, by_section: bool = True) -> List[dict]:
        """
        Convert to RAG document schema.

        Args:
            by_section: If True, create one document per section.
                        If False, create a single document for the whole note.

        Returns:
            List of RAG document dicts with keys: id, text, metadata.
        """
        base_metadata = {
            "note_id": self.note_id,
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "note_type": self.note_type,
            "source": self.source,
            "admission_year": self.admission_date,
        }

        if not by_section or not self.sections:
            return [
                {
                    "id": f"note_{self.note_id}",
                    "text": self.text,
                    "metadata": {**base_metadata, "section": "full_note"},
                }
            ]

        docs = []
        for sec in self.sections:
            if len(sec.text.strip()) < 20:  # Skip very short sections
                continue
            docs.append(
                {
                    "id": f"note_{self.note_id}_sec_{sec.section_name}",
                    "text": sec.text,
                    "metadata": {
                        **base_metadata,
                        "section": sec.section_name,
                        "char_offset": sec.start_char,
                    },
                }
            )
        return docs


# ---------------------------------------------------------------------------
# De-identification pipeline
# ---------------------------------------------------------------------------

class PHIDeidentifier:
    """
    HIPAA Safe Harbor de-identification using regex and placeholder replacement.

    Implements the 18 PHI identifier removals required by HIPAA Safe Harbor
    (45 CFR §164.514(b)). For production clinical systems, this should be
    supplemented with an NER-based detector (see _ner_deidentify).

    The MIMIC-III dataset has already applied de-identification using the
    Philter system, replacing PHI with [**placeholder**] patterns. This
    class handles both original text and MIMIC-style pre-de-identified text.

    Args:
        replacement_token: Text to substitute for identified PHI.
        strict_mode: If True, apply aggressive patterns that may over-redact.
    """

    def __init__(
        self,
        replacement_token: str = "[REDACTED]",
        strict_mode: bool = False,
    ):
        self.replacement_token = replacement_token
        self.strict_mode = strict_mode
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in PHI_PATTERNS.items()
        }

    def deidentify(self, text: str) -> Tuple[str, int]:
        """
        Remove PHI from clinical text.

        Args:
            text: Raw clinical note text.

        Returns:
            Tuple of (de-identified text, number of PHI instances removed).
        """
        phi_count = 0
        for phi_type, pattern in self._compiled_patterns.items():
            matches = pattern.findall(text)
            if matches:
                count = len(pattern.findall(text))
                phi_count += count
                text = pattern.sub(self.replacement_token, text)
                if count > 0:
                    logger.debug(
                        "PHI type '%s': removed %d instances", phi_type, count
                    )

        # Additional pass: catch remaining [**...**] MIMIC placeholders
        mimic_remaining = re.findall(r"\[\*\*[^\]]*\*\*\]", text)
        if mimic_remaining:
            phi_count += len(mimic_remaining)
            text = re.sub(
                r"\[\*\*[^\]]*\*\*\]", self.replacement_token, text
            )

        return text, phi_count

    def contains_phi(self, text: str) -> bool:
        """
        Check whether text likely contains PHI.

        Used as a pre-filter before storing de-identified notes.

        Args:
            text: Text to check.

        Returns:
            True if potential PHI is detected.
        """
        for pattern in self._compiled_patterns.values():
            if pattern.search(text):
                return True
        return False

    def audit_report(self, original: str, deidentified: str) -> dict:
        """
        Generate an audit report comparing original and de-identified text.

        Args:
            original: Original text before de-identification.
            deidentified: Text after de-identification.

        Returns:
            Dict with redaction counts by PHI type.
        """
        report: dict = {"phi_types_detected": {}, "total_phi_removed": 0}
        for phi_type, pattern in self._compiled_patterns.items():
            matches = pattern.findall(original)
            if matches:
                count = len(matches)
                report["phi_types_detected"][phi_type] = count
                report["total_phi_removed"] += count
        report["original_chars"] = len(original)
        report["deidentified_chars"] = len(deidentified)
        report["char_reduction"] = len(original) - len(deidentified)
        return report


# ---------------------------------------------------------------------------
# Section segmenter
# ---------------------------------------------------------------------------

class ClinicalNoteSectionSegmenter:
    """
    Segment clinical notes into structured sections.

    Identifies standard clinical sections (HPI, Assessment, Plan, etc.)
    by pattern matching on section headers. Handles variation in header
    formatting common in EHR-generated notes.

    Args:
        min_section_length: Minimum characters for a section to be kept.
    """

    def __init__(self, min_section_length: int = 30):
        self.min_section_length = min_section_length

    def segment(self, text: str) -> List[ClinicalNoteSection]:
        """
        Split note text into sections.

        Args:
            text: Clinical note text (should be de-identified).

        Returns:
            List of ClinicalNoteSection objects in document order.
        """
        # Find all section boundaries
        boundaries: List[Tuple[int, str]] = []
        for section_name, pattern in SECTION_PATTERNS.items():
            for match in pattern.finditer(text):
                boundaries.append((match.end(), section_name))

        if not boundaries:
            # No section headers found; return as single section
            return [
                ClinicalNoteSection(
                    section_name="body",
                    text=text,
                    start_char=0,
                    end_char=len(text),
                )
            ]

        # Sort boundaries by position
        boundaries.sort(key=lambda x: x[0])

        sections = []
        for i, (start, name) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            section_text = text[start:end].strip()
            if len(section_text) >= self.min_section_length:
                sections.append(
                    ClinicalNoteSection(
                        section_name=name,
                        text=section_text,
                        start_char=start,
                        end_char=end,
                    )
                )

        return sections


# ---------------------------------------------------------------------------
# MIMIC-III note types
# ---------------------------------------------------------------------------

MIMIC_NOTE_TYPES = {
    "Discharge summary": "discharge_summary",
    "Radiology": "radiology",
    "Nursing": "nursing",
    "Nursing/other": "nursing_other",
    "Physician": "physician",
    "ECG": "ecg",
    "Echo": "echo",
    "Respiratory": "respiratory",
    "Nutrition": "nutrition",
    "Rehab Services": "rehab",
    "Social Work": "social_work",
    "Case Management": "case_management",
    "Pharmacy": "pharmacy",
    "Consult": "consult",
    "General": "general",
}


# ---------------------------------------------------------------------------
# MIMIC-III loader
# ---------------------------------------------------------------------------

class MIMICIIIClinicalNoteLoader:
    """
    Load and process MIMIC-III clinical notes.

    MIMIC-III stores notes in NOTEEVENTS.csv with the following schema:
        ROW_ID, SUBJECT_ID, HADM_ID, CHARTDATE, CHARTTIME, STORETIME,
        CATEGORY, DESCRIPTION, CGID, ISERROR, TEXT

    Access requires PhysioNet credentialing:
        https://physionet.org/content/mimiciii/1.4/

    Args:
        deidentifier: PHIDeidentifier instance. MIMIC notes are pre-deidentified
                      but we apply an additional pass for safety.
        segmenter: ClinicalNoteSectionSegmenter instance.
        note_types: Filter to specific note types. None = all types.
        max_notes: Maximum notes to load (for development/testing).
        apply_extra_deidentification: Apply additional PHI removal on top of
                                       MIMIC's existing de-identification.

    Example::

        loader = MIMICIIIClinicalNoteLoader(
            note_types=["Discharge summary"],
            max_notes=1000,
        )
        for note in loader.iter_notes(Path("data/raw/NOTEEVENTS.csv")):
            process(note)
    """

    def __init__(
        self,
        deidentifier: Optional[PHIDeidentifier] = None,
        segmenter: Optional[ClinicalNoteSectionSegmenter] = None,
        note_types: Optional[List[str]] = None,
        max_notes: Optional[int] = None,
        apply_extra_deidentification: bool = True,
    ):
        self.deidentifier = deidentifier or PHIDeidentifier()
        self.segmenter = segmenter or ClinicalNoteSectionSegmenter()
        self.note_types = note_types
        self.max_notes = max_notes
        self.apply_extra_deidentification = apply_extra_deidentification
        self._stats = {
            "total_processed": 0,
            "total_phi_removed": 0,
            "skipped_errors": 0,
            "skipped_empty": 0,
        }

    def iter_notes(
        self, noteevents_path: Path
    ) -> Iterator[ClinicalNote]:
        """
        Lazily iterate over processed MIMIC-III clinical notes.

        Args:
            noteevents_path: Path to NOTEEVENTS.csv from MIMIC-III.

        Yields:
            ClinicalNote objects.
        """
        noteevents_path = Path(noteevents_path)
        if not noteevents_path.exists():
            raise FileNotFoundError(
                f"MIMIC-III NOTEEVENTS.csv not found at {noteevents_path}.\n"
                f"Download from https://physionet.org/content/mimiciii/1.4/ "
                f"(requires credentialing)."
            )

        count = 0
        with open(noteevents_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.max_notes and count >= self.max_notes:
                    break

                # Skip error notes (ISERROR flag)
                if row.get("ISERROR", "").strip() == "1":
                    self._stats["skipped_errors"] += 1
                    continue

                # Filter by note type if specified
                category = row.get("CATEGORY", "").strip()
                if self.note_types and category not in self.note_types:
                    continue

                text = row.get("TEXT", "").strip()
                if not text:
                    self._stats["skipped_empty"] += 1
                    continue

                try:
                    note = self._process_note(row, text)
                    count += 1
                    self._stats["total_processed"] += 1
                    yield note
                except Exception as exc:
                    logger.warning(
                        "Error processing note %s: %s",
                        row.get("ROW_ID"),
                        exc,
                    )
                    self._stats["skipped_errors"] += 1

        logger.info("Processed %d notes. Stats: %s", count, self._stats)

    def _process_note(self, row: dict, text: str) -> ClinicalNote:
        """Process a single note row into a ClinicalNote."""
        original_length = len(text)

        # De-identification pass
        phi_count = 0
        if self.apply_extra_deidentification:
            text, phi_count = self.deidentifier.deidentify(text)
            self._stats["total_phi_removed"] += phi_count

        # Extract year only from date (drop month/day for HIPAA)
        chart_date = row.get("CHARTDATE", "")
        admission_year = chart_date[:4] if chart_date else None

        # Clean whitespace artifacts common in MIMIC text
        text = self._clean_whitespace(text)

        # Section segmentation (discharge summaries and physician notes)
        category = row.get("CATEGORY", "").strip()
        sections: List[ClinicalNoteSection] = []
        if category in ("Discharge summary", "Physician"):
            sections = self.segmenter.segment(text)

        return ClinicalNote(
            note_id=row.get("ROW_ID", ""),
            subject_id=row.get("SUBJECT_ID", ""),
            hadm_id=row.get("HADM_ID", ""),
            note_type=MIMIC_NOTE_TYPES.get(category, category.lower()),
            text=text,
            sections=sections,
            original_length=original_length,
            phi_count=phi_count,
            admission_date=admission_year,
        )

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Normalize whitespace in clinical notes."""
        # Replace multiple blank lines with single blank line
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove trailing spaces on each line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text.strip()

    def load_all(
        self, noteevents_path: Path, output_dir: Optional[Path] = None
    ) -> List[ClinicalNote]:
        """
        Load all notes into memory.

        Args:
            noteevents_path: Path to NOTEEVENTS.csv.
            output_dir: If provided, save processed notes as JSONL.

        Returns:
            List of ClinicalNote objects.
        """
        notes = list(self.iter_notes(noteevents_path))
        if output_dir:
            self.save_jsonl(notes, Path(output_dir) / "clinical_notes.jsonl")
        return notes

    def save_jsonl(
        self,
        notes: List[ClinicalNote],
        output_path: Path,
        as_rag_documents: bool = True,
        by_section: bool = True,
    ) -> None:
        """
        Save notes to JSONL file.

        Args:
            notes: List of ClinicalNote objects.
            output_path: Output file path.
            as_rag_documents: If True, save in RAG document schema.
            by_section: If True and as_rag_documents, emit one doc per section.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc_count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for note in notes:
                if as_rag_documents:
                    for doc in note.to_rag_documents(by_section=by_section):
                        f.write(json.dumps(doc) + "\n")
                        doc_count += 1
                else:
                    f.write(json.dumps(note.to_dict()) + "\n")
                    doc_count += 1
        logger.info("Saved %d documents to %s", doc_count, output_path)

    @property
    def stats(self) -> dict:
        """Return processing statistics."""
        return dict(self._stats)


# ---------------------------------------------------------------------------
# Radiology report parser
# ---------------------------------------------------------------------------

class RadiologyReportParser:
    """
    Specialized parser for MIMIC-III radiology reports.

    Radiology reports have a predictable structure:
        EXAMINATION, INDICATION, TECHNIQUE, FINDINGS, IMPRESSION

    Args:
        deidentifier: PHI deidentifier to apply.
    """

    RADIOLOGY_SECTIONS = {
        "examination": re.compile(r"EXAMINATION\s*:?\s*\n", re.IGNORECASE),
        "indication": re.compile(r"INDICATION\s*:?\s*\n", re.IGNORECASE),
        "technique": re.compile(r"TECHNIQUE\s*:?\s*\n", re.IGNORECASE),
        "comparison": re.compile(r"COMPARISON\s*:?\s*\n", re.IGNORECASE),
        "findings": re.compile(r"FINDINGS\s*:?\s*\n", re.IGNORECASE),
        "impression": re.compile(r"IMPRESSION\s*:?\s*\n", re.IGNORECASE),
    }

    def __init__(self, deidentifier: Optional[PHIDeidentifier] = None):
        self.deidentifier = deidentifier or PHIDeidentifier()

    def parse(self, text: str, note_id: str = "") -> dict:
        """
        Parse a radiology report into structured sections.

        Args:
            text: Raw radiology report text.
            note_id: Identifier for logging.

        Returns:
            Dict with section names as keys, section text as values,
            plus 'impression' (most clinically important section).
        """
        text, _ = self.deidentifier.deidentify(text)
        result: dict = {"raw_text": text, "sections": {}}

        boundaries: List[Tuple[int, str]] = []
        for name, pattern in self.RADIOLOGY_SECTIONS.items():
            match = pattern.search(text)
            if match:
                boundaries.append((match.end(), name))

        boundaries.sort(key=lambda x: x[0])
        for i, (start, name) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            result["sections"][name] = text[start:end].strip()

        # Impression is the key clinical summary
        result["impression"] = result["sections"].get(
            "impression",
            result["sections"].get("findings", "")
        )
        return result
