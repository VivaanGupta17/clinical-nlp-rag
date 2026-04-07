# ClinicalRAG System Architecture

## Overview

ClinicalRAG is a production-grade Retrieval-Augmented Generation (RAG) system for biomedical and clinical text. It is designed around three core design principles:

1. **Safety first**: Hallucinations in clinical settings can cause patient harm. Every architectural decision prioritizes faithfulness verification.
2. **Domain specialization**: Biomedical text requires specialized models (PubMedBERT, ScispaCy), not general-purpose ones.
3. **Modularity**: Each component is independently replaceable, testable, and configurable via YAML.

---

## High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         ClinicalRAG System                                 │
│                                                                             │
│  ┌─────────────┐  OFFLINE PIPELINE (Index Build Time)  ┌───────────────┐  │
│  │             │                                         │               │  │
│  │   PubMed    │──→ Ingestion ──→ Chunking ──→         │   FAISS IVF   │  │
│  │  Entrez API │                             Embedding  │   Dense Index │  │
│  │             │   XML parse    Sentence /   PubMedBERT │               │  │
│  └─────────────┘   NER extract  Semantic /   GPU batch  │   BM25        │  │
│                                 Section     encoding    │   Sparse Index│  │
│  ┌─────────────┐                                        │               │  │
│  │  MIMIC-III  │──→ De-ID ─────→ Section ──→           └───────┬───────┘  │
│  │  Clinical   │    PHI remove   Segment                        │          │
│  │  Notes      │    Audit log    HPI/Assess                     │          │
│  └─────────────┘                                                │          │
│                                                                 │          │
│  ┌─────────────────────────────────────────────────────────────▼───────┐  │
│  │                    ONLINE PIPELINE (Query Time)                      │  │
│  │                                                                       │  │
│  │  User Query ──→ UMLS Expansion ──→ Hybrid Retrieval (RRF)           │  │
│  │                   Synonyms          Dense + BM25                     │  │
│  │                                          │                           │  │
│  │                                          ▼                           │  │
│  │                                  Cross-Encoder Re-rank               │  │
│  │                                  (MedCPT / DeBERTa)                  │  │
│  │                                          │                           │  │
│  │                                          ▼                           │  │
│  │                                    Top-K Context                     │  │
│  │                                          │                           │  │
│  │                                          ▼                           │  │
│  │                                   LLM Generation                     │  │
│  │                                   GPT-4 / BioGPT                     │  │
│  │                                          │                           │  │
│  │                                          ▼                           │  │
│  │                              Hallucination Detection                 │  │
│  │                              Claim Extraction + NLI                  │  │
│  │                              Faithfulness Score                      │  │
│  │                                          │                           │  │
│  │                                          ▼                           │  │
│  │                              Cited Answer + Warning                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Data Ingestion Layer (`src/ingestion/`)

#### PubMedLoader (`pubmed_loader.py`)

Handles PubMed data retrieval via NCBI Entrez API:

```
EntrezClient  ──────────────────────────────→  NCBI Servers
    │                                             (eSearch → efetch)
    │  XML response
    ▼
PubMedXMLParser
    │  PubMedAbstract objects
    ▼
PubMedCache ← disk cache (gzip, 72h TTL)
    │
    ▼
PubMedAbstract → .to_rag_document() → {id, text, metadata}
```

**Design decision**: Using history server (`usehistory=y`) for large queries avoids transmitting thousands of PMIDs in URL parameters.

**Rate limiting**: 0.34s between requests without API key (3 req/s), 0.1s with key (10 req/s). Implemented as wall-clock time tracking, not thread sleep, to avoid blocking.

#### MIMICIIIClinicalNoteLoader (`clinical_note_loader.py`)

Processes MIMIC-III NOTEEVENTS.csv with PHI safety:

```
NOTEEVENTS.csv (CSV reader, streaming)
    │
    ▼ per row
PHIDeidentifier.deidentify()
    │  (18-pattern regex cascade)
    │  → audit: phi_count removed
    ▼
ClinicalNoteSectionSegmenter.segment()
    │  (header pattern matching)
    │  → HPI / Assessment / Plan sections
    ▼
ClinicalNote → .to_rag_documents(by_section=True)
    → [{id, text, metadata: {section, note_type, ...}}]
```

**Memory efficiency**: Streaming CSV processing with `iter_notes()` avoids loading all 2.1M MIMIC notes into RAM.

---

### 2. Chunking Layer (`src/chunking/`)

Three complementary strategies:

| Strategy | Algorithm | Best For | Chunk Coherence |
|---|---|---|---|
| SentenceChunker | BiomedAbbrev tokenizer + sliding window | PubMed abstracts | Syntactic |
| SemanticChunker | Embed → cosine similarity → breakpoints | Long clinical notes | Semantic |
| SectionAwareChunker | Preserve section boundaries | MIMIC discharge summaries | Structural |

**Semantic chunking algorithm**:
1. Split text into sentences using BiomedicalSentenceSplitter (handles "t.i.d.", "Dr.", etc.)
2. Embed each sentence with the configured biomedical embedder
3. Compute adjacent cosine similarities: `sim[i] = cos(emb[i], emb[i+1])`
4. Find breakpoints at similarity < percentile(similarities, 100 - breakpoint_percentile)
5. Group sentences into coherent chunks at breakpoints

**Overlap**: Sentence-level chunker maintains a rolling window of the previous N tokens. This ensures context continuity at chunk boundaries — critical for answering questions that span sentence groups.

---

### 3. Embedding Layer (`src/embeddings/`)

#### Model Selection Rationale

| Model | Training Data | Why Use |
|---|---|---|
| PubMedBERT | 21M PubMed abstracts | Best general biomedical embedding |
| MedCPT | PubMed search logs (query-article pairs) | Best for retrieval (asymmetric) |
| BioClinicalBERT | MIMIC-III clinical notes | Best for clinical text |
| SPECTER2 | Citation graphs | Best for paper similarity |

**Asymmetric encoding**: MedCPT uses separate query and article encoders trained on actual PubMed search interactions. Queries and documents have different statistical properties; separate encoders capture this.

#### Embedding Cache Architecture

```
encode(texts) 
    │
    ├─→ cache.get_batch(model_name, texts)  →  Cache HIT: return saved
    │
    └─→ cache MISS list
            │
            ▼
        _encode_batch() [GPU forward pass]
            │  mean pooling over token embeddings
            │  L2 normalization
            ▼
        cache.set_batch(model_name, texts, embeddings)
```

MD5 hash of `(model_name, text)` used as cache key. Gzip-compressed `.npy` files.

---

### 4. Vector Store Layer (`src/vectorstore/`)

#### FAISS Index Types

| Type | Algorithm | Training Required | Memory | Speed | Accuracy |
|---|---|---|---|---|---|
| `flat` | Exact exhaustive search | No | High | O(N) | Perfect |
| `ivf` | Inverted file + quantizer | Yes (N ≥ nlist) | Medium | O(nprobe/nlist × N) | ~95-99% |
| `hnsw` | Hierarchical navigable small world | No | High | O(log N) | ~95-99% |

**Rule of thumb**: `flat` for <100K docs, `ivf` for 100K–10M, `hnsw` for real-time serving (lowest latency).

#### BM25 Implementation

Standard Okapi BM25 with clinical stopword list:
```
score(d, q) = Σ_t∈q  IDF(t) × (tf(t,d) × (k1+1)) / (tf(t,d) + k1×(1-b+b×dl/avgdl))

Where:
  k1 = 1.5  (term frequency saturation)
  b  = 0.75 (length normalization)
  IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

Clinical stopwords ("patient", "history", "treatment") are removed to prevent uninformative high-frequency matches.

#### Reciprocal Rank Fusion

```
RRF_score(d) = dense_weight × 1/(k + rank_dense(d))
             + sparse_weight × 1/(k + rank_sparse(d))

Where k=60 (Cormack et al. 2009), dense_weight=0.7, sparse_weight=0.3
```

RRF is score-normalized by construction — no calibration needed between BM25 (unnormalized) and cosine similarity ([-1, 1]).

---

### 5. Retrieval Layer (`src/retrieval/`)

#### Query Processing Pipeline

```
User Query
    │
    ▼ (optional)
UMLSQueryExpander
    │  "heart attack" → "heart attack myocardial infarction MI STEMI"
    ▼
BiomedicalEmbedder.encode_queries()
    │  (768-dim embedding)
    ▼
HybridIndex.search(query_text, query_embedding)
    │  dense top-50 + sparse top-50 → RRF → top-50
    ▼ (optional)
CrossEncoderReranker.rerank()
    │  cross-encoder scores top-50 → re-rank → top-10
    ▼
List[RetrievalResult]
```

**Re-ranking trade-off**: Cross-encoder improves P@5 by ~8% but adds ~150ms latency for 50 candidates. For interactive use, disable re-ranking (--no-rerank). For batch evaluation, always enable it.

---

### 6. Generation Layer (`src/generation/`)

#### Prompt Engineering for Clinical Safety

Clinical prompts include explicit safety constraints:
1. "Answer ONLY based on the provided context"
2. "If context is insufficient, say so explicitly"
3. "For drug dosages, ALWAYS cite the specific source"
4. "Acknowledge uncertainty when evidence is conflicting"

These constraints are enforced at the prompt level, then verified at the faithfulness level.

#### Citation Mechanism

The [Doc N] citation scheme works as follows:
1. Context documents are numbered [Doc 1] through [Doc N] in the prompt
2. The LLM is instructed to cite sources using [Doc N] notation
3. At generation time, numbered references are mapped to actual PMIDs/note IDs
4. Citations are extracted and included in the `GeneratedAnswer.citations` list

#### Streaming Architecture

```python
# Token stream via Server-Sent Events (SSE)
for token in generator.generate_stream(query, docs):
    yield f"data: {token}\n\n"
```

SSE-compatible streaming enables real-time display in web interfaces without full response buffering.

---

### 7. Hallucination Detection Layer (`src/generation/`)

#### Three-Layer Architecture

```
Generated Answer
    │
    ▼ Layer 1: Claim Extraction
AtomicClaimExtractor (LLM-based or heuristic)
    │  "Metformin lowers blood glucose" (claim 1)
    │  "by activating AMPK" (claim 2)
    │  "reducing hepatic glucose production" (claim 3)
    ▼ Layer 2: NLI Entailment Verification
NLIEntailmentVerifier (DeBERTa-NLI)
    │  For each claim × each context passage:
    │  P(entailment) > 0.7 → "supported"
    │  P(contradiction) > 0.5 → "contradicted"
    │  Otherwise → "neutral"
    ▼ Layer 3: Report Generation
FaithfulnessReport
    │  faithfulness_score = supported/total_claims
    │  hallucination_rate = (total - supported)/total
    │  citation_accuracy = cited_passages_with_support/total_citations
```

**Calibration**: The 0.7 entailment threshold is intentionally conservative for clinical text. Better to flag a true fact as uncertain than to miss a hallucinated claim.

---

### 8. NLP Layer (`src/ner/`)

#### Entity Extraction Pipeline

```
Clinical Text
    │
    ├─→ ScispaCyNER (en_ner_bc5cdr_md)
    │   Drug + Disease entities
    │   Spacy dependency parse
    │
    ├─→ BERTBiomedicalNER (d4data/biomedical-ner-all)
    │   All entity types
    │   Token-level classification
    │
    ▼ Ensemble
BiomedicalNER._deduplicate()
    │  Span conflict resolution: keep highest confidence
    ▼
List[BiomedicalEntity] → UMLS linking (optional)
    │  CUI lookup in UMLS KB
    ▼
NERResult {drugs, diseases, genes, ...}
```

#### Relation Extraction Architecture

Pipeline RE (entity pair enumeration + classification):
```
NERResult
    │
    ▼ DDI: for each drug pair in sentence
DrugDDIExtractor
    │  Marked sentence: "warfarin [DRUG1] ... fluconazole [DRUG2]"
    │  → DDI classifier → "mechanism" (confidence 0.89)
    ▼
    + DrugADRExtractor (regex patterns + NER context)
    + GeneDiseaseExtractor (regex + BERT classifier)
    ▼
List[BiologicalRelation]
```

---

## Scalability Considerations

| Component | 10K docs | 100K docs | 1M docs |
|---|---|---|---|
| Embedding | ~10min (CPU) / ~2min (GPU) | ~100min CPU / ~20min GPU | Requires GPU cluster |
| FAISS type | flat | ivf (nlist=300) | ivf (nlist=1000) or HNSW |
| Index size | ~60MB | ~600MB | ~6GB |
| Query latency | 5ms | 50ms | 200ms |
| Memory needed | 1GB | 8GB | 80GB |

---

## Configuration Reference

All system behavior is controlled via `configs/pubmed_rag_config.yaml`. Key settings:

| Section | Key Parameters |
|---|---|
| `embedding.model_id` | Switches between PubMedBERT, MedCPT, etc. |
| `index.type` | `hybrid`, `faiss_flat`, `faiss_ivf`, `faiss_hnsw` |
| `retrieval.use_reranker` | Enable/disable cross-encoder (speed vs quality) |
| `retrieval.use_query_expansion` | Enable/disable UMLS synonym expansion |
| `generation.provider` | `openai`, `azure_openai`, `huggingface` |
| `hallucination.faithfulness_threshold` | Alert threshold for low-faithfulness answers |
| `hipaa.audit_logging` | Enable HIPAA-compliant access logging |

---

## Testing Architecture

```
tests/
├── unit/
│   ├── test_deidentifier.py    # PHI removal correctness
│   ├── test_chunker.py         # Chunking edge cases
│   ├── test_embedder.py        # Embedding shape/norm tests
│   ├── test_bm25.py            # BM25 scoring correctness
│   └── test_hallucination.py   # Claim extraction + NLI tests
├── integration/
│   ├── test_ingestion.py       # PubMed API (mocked)
│   ├── test_retrieval.py       # End-to-end retrieval
│   └── test_generation.py     # Full pipeline (mocked LLM)
└── conftest.py                  # Shared fixtures
```

Run tests: `pytest tests/ -v --cov=src/ --cov-report=html`
