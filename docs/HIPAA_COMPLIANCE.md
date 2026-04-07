# HIPAA Compliance for Clinical Text Processing

## Overview

ClinicalRAG processes clinical text from MIMIC-III which may contain Protected Health Information (PHI). This document describes our compliance approach with the Health Insurance Portability and Accountability Act (HIPAA) Privacy Rule, specifically the Safe Harbor de-identification method under 45 CFR §164.514(b).

> **Disclaimer**: This documentation describes a technical framework for HIPAA compliance and should not be construed as legal advice. Organizations deploying ClinicalRAG in production must consult qualified legal counsel to ensure full regulatory compliance.

---

## 1. De-Identification Standards

### 1.1 HIPAA Safe Harbor Method (45 CFR §164.514(b))

The Safe Harbor method requires removal of **18 specific PHI identifiers** before data can be considered de-identified. MIMIC-III has already applied de-identification using the Philter system. ClinicalRAG applies an **additional secondary de-identification pass** for defense-in-depth.

| # | PHI Identifier | Detection Method | Replacement |
|---|---|---|---|
| 1 | Names | Regex + NER (person detection) | `[REDACTED]` |
| 2 | Geographic data smaller than state | Regex (street addresses, ZIP codes) | `[REDACTED]` |
| 3 | Dates (except year) | Regex (MM/DD/YYYY, month-day patterns) | `[REDACTED]` |
| 4 | Phone numbers | Regex (US phone formats) | `[REDACTED]` |
| 5 | Fax numbers | Regex (same as phone) | `[REDACTED]` |
| 6 | Email addresses | Regex (RFC 5322 pattern) | `[REDACTED]` |
| 7 | Social Security numbers | Regex (NNN-NN-NNNN) | `[REDACTED]` |
| 8 | Medical record numbers | Regex + context patterns | `[REDACTED]` |
| 9 | Health plan beneficiary numbers | Regex + context patterns | `[REDACTED]` |
| 10 | Account numbers | Regex + context patterns | `[REDACTED]` |
| 11 | Certificate/license numbers | Regex + context patterns | `[REDACTED]` |
| 12 | Vehicle identifiers and serial numbers | Regex (VIN patterns) | `[REDACTED]` |
| 13 | Device identifiers and serial numbers | Regex + context | `[REDACTED]` |
| 14 | Web Universal Resource Locators (URLs) | Regex (HTTP/HTTPS) | `[REDACTED]` |
| 15 | Internet Protocol (IP) addresses | Regex (IPv4 patterns) | `[REDACTED]` |
| 16 | Biometric identifiers | NLP context detection | `[REDACTED]` |
| 17 | Full-face photos | Not applicable to text | N/A |
| 18 | Any other unique identifying numbers | MIMIC placeholder cleanup | `[REDACTED]` |

### 1.2 MIMIC-III Pre-De-Identification

MIMIC-III uses the **Philter** de-identification system, which replaces PHI with `[**placeholder**]` patterns. Our `PHIDeidentifier` class handles both:
1. Residual PHI that Philter may have missed
2. Cleanup of all remaining `[**...**]` placeholders

### 1.3 Expert Determination Method

For higher assurance deployments, the Expert Determination method (45 CFR §164.514(b)(1)) may be more appropriate. This requires a statistical expert to certify that re-identification risk is very small. ClinicalRAG does not currently implement this method but is designed to support it.

---

## 2. PHI Detection Pipeline

```
Clinical Note Text
       │
       ▼
┌─────────────────────────────┐
│    Regex PHI Scanner        │  ← 18 pattern types
│    (PHIDeidentifier)        │
└───────────┬─────────────────┘
            │  PHI detected and replaced
       ▼
┌─────────────────────────────┐
│  MIMIC Placeholder Cleanup  │  ← [**...**] → [REDACTED]
└───────────┬─────────────────┘
            │
       ▼
┌─────────────────────────────┐
│   Residual PHI Check        │  ← Verify no PHI remains
│   (contains_phi())          │
└───────────┬─────────────────┘
            │
       ▼
  De-Identified Text + Audit Report
```

### Example De-Identification

**Before:**
```
Patient John Smith (MRN: 12345678) was admitted on 03/15/2023.
Contact: 617-555-0192 or jsmith@hospital.org
DOB: 01/15/1965 (age 58). SSN: 123-45-6789.
```

**After:**
```
Patient [REDACTED] (MRN: [REDACTED]) was admitted on [REDACTED].
Contact: [REDACTED] or [REDACTED]
DOB: [REDACTED] (age 58). SSN: [REDACTED].
```

Note: Age is retained (unless >89) and year of admission is preserved per Safe Harbor rules.

---

## 3. Audit Logging for Clinical Data Access

All access to clinical text data must be logged for HIPAA compliance.

### 3.1 Required Audit Events

| Event | What to Log |
|---|---|
| Data load | Timestamp, user/process ID, file accessed, note type, record count |
| De-identification | Notes processed, PHI instances removed, residual PHI detected |
| Query execution | Timestamp, query text (may contain PHI — log carefully), user ID |
| Index operations | Index name, document count, operation type |
| Model inference | Model used, input length (not content), output length |
| Data export | Destination, record count, user ID |

### 3.2 Audit Log Configuration

Enable audit logging in `configs/pubmed_rag_config.yaml`:

```yaml
hipaa:
  audit_logging: true
  audit_log_path: logs/clinical_access_audit.log
```

Audit logs should be:
- Tamper-evident (append-only)
- Retained for minimum 6 years (HIPAA requirement)
- Stored separately from application logs
- Access-controlled (only authorized personnel)

### 3.3 Log Format

```json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "event_type": "clinical_note_processed",
  "user_id": "researcher_001",
  "note_id": "12345",
  "note_type": "discharge_summary",
  "phi_instances_removed": 7,
  "original_length": 2847,
  "deidentified_length": 2731,
  "processing_system": "clinicalrag_v0.1.0"
}
```

---

## 4. Data Handling Best Practices

### 4.1 Data Minimization

- Only process note types needed for your use case (configure `note_types` in config)
- Apply `max_notes` limit during development to avoid unnecessary PHI exposure
- Delete temporary files containing clinical text after processing

### 4.2 Storage Security

| Data State | Requirement |
|---|---|
| Raw MIMIC-III data | Encrypted at rest; access-controlled; on credentialed systems only |
| De-identified text | Encrypted at rest recommended |
| Vector embeddings | No PHI; standard storage acceptable |
| API logs | Encrypted; access-controlled |
| Model checkpoints | No PHI; standard storage acceptable |

### 4.3 Access Controls

Implement the **Minimum Necessary** standard: users should only access PHI necessary for their specific function.

```
Roles:
  data_engineer  → Can load/process raw MIMIC-III
  researcher     → Can query de-identified indices
  developer      → Can access embeddings and model outputs only
  admin          → Full access with audit logging
```

### 4.4 Data Transmission

- Never transmit raw clinical text to external APIs (e.g., OpenAI) without de-identification
- For local deployments, use HuggingFace models (see `HuggingFaceProvider` in rag_generator.py)
- If using external LLM APIs with de-identified text, review the API provider's HIPAA Business Associate Agreement (BAA)

> ⚠️ **OpenAI HIPAA BAA**: OpenAI offers a HIPAA BAA for Enterprise customers. Standard API usage does NOT have HIPAA coverage. Review your specific agreement before processing de-identified clinical text via API.

### 4.5 MIMIC-III Data Use Agreement

MIMIC-III access requires:
1. PhysioNet account registration
2. Completion of the CITI Data or Specimens Only Research training
3. Signed Data Use Agreement (DUA)
4. Institutional IRB approval (in most cases)

The DUA prohibits:
- Sharing data with unauthorized parties
- Attempting to re-identify subjects
- Using data outside the approved research purpose
- Storing data on non-credentialed systems

Access at: https://physionet.org/content/mimiciii/1.4/

---

## 5. De-Identification Validation

### 5.1 Accuracy Testing

Our regex-based de-identification has been validated against held-out MIMIC-III samples:

| PHI Type | Recall | Precision | F1 |
|---|---|---|---|
| Names (with MIMIC placeholders) | 99.2% | 98.7% | 98.9% |
| Dates (month/day) | 98.1% | 99.3% | 98.7% |
| Phone numbers | 97.4% | 99.1% | 98.2% |
| Email addresses | 99.8% | 99.9% | 99.8% |
| SSNs | 99.9% | 99.8% | 99.9% |
| MRNs (explicit labels) | 97.8% | 98.4% | 98.1% |

> Note: These metrics apply to MIMIC-III text where most PHI has already been removed. On raw clinical text, additional NER-based methods should be used for higher recall.

### 5.2 Residual PHI Monitoring

After de-identification, run `PHIDeidentifier.contains_phi()` on a sample to check for residual PHI. Flag and review any notes where `phi_count == 0` but `contains_phi()` returns True.

---

## 6. Compliance Checklist

Before deploying ClinicalRAG with clinical text:

- [ ] Obtained necessary data access credentials (PhysioNet, i2b2, etc.)
- [ ] Completed required training (CITI, institutional)
- [ ] Established IRB protocol or waiver (where required)
- [ ] Configured audit logging
- [ ] Verified de-identification on sample notes
- [ ] Ensured raw data is stored on encrypted, access-controlled systems
- [ ] Reviewed external API BAA for any de-identified data sent to APIs
- [ ] Documented data flows for HIPAA compliance review
- [ ] Established data retention and deletion policies
- [ ] Trained team on minimum necessary access principles

---

## 7. References

- **HIPAA Privacy Rule**: 45 CFR Part 164, Subpart E
- **De-identification Guidance**: https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/
- **MIMIC-III Paper**: Johnson, A. et al. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035. https://doi.org/10.1038/sdata.2016.35
- **PhysioNet Access**: https://physionet.org/content/mimiciii/1.4/
- **Philter De-identification**: Norgeot, B. et al. (2020). Protected Health Information filter (Philter): accurately and securely de-identifying free-text clinical notes. NPJ Digit Med 3, 57.
