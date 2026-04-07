#!/usr/bin/env python3
"""
Data ingestion script for ClinicalRAG pipeline.

Downloads and processes data from PubMed and/or MIMIC-III,
then saves chunked documents ready for embedding and indexing.

Usage:
    # PubMed by MeSH term
    python scripts/ingest.py --source pubmed --mesh-terms "Atrial Fibrillation" --max-results 1000

    # PubMed by category (downloads multiple MeSH terms)
    python scripts/ingest.py --source pubmed --category cardiology --max-per-term 2000

    # MIMIC-III clinical notes
    python scripts/ingest.py --source mimic --noteevents-path data/raw/NOTEEVENTS.csv \\
        --note-types "Discharge summary" --max-notes 10000

    # Both sources
    python scripts/ingest.py --source both --category ai_clinical

    # With semantic chunking (requires embedding model)
    python scripts/ingest.py --source pubmed --mesh-terms "COVID-19" --chunking semantic
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Adjust import path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pubmed_loader import PubMedBatchDownloader, PubMedLoader
from src.ingestion.clinical_note_loader import MIMICIIIClinicalNoteLoader
from src.chunking.semantic_chunker import ChunkerFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ClinicalRAG data ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["pubmed", "mimic", "both"],
        default="pubmed",
        help="Data source to ingest",
    )
    parser.add_argument(
        "--config",
        default="configs/pubmed_rag_config.yaml",
        help="Config file path",
    )
    # PubMed options
    parser.add_argument(
        "--mesh-terms",
        nargs="+",
        help="MeSH terms to download",
    )
    parser.add_argument(
        "--category",
        choices=list(PubMedBatchDownloader.CLINICAL_DOMAINS.keys()),
        help="Predefined clinical domain category",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=1000,
        help="Maximum PubMed results per term",
    )
    parser.add_argument(
        "--query",
        help="Free-text Entrez query string",
    )
    # MIMIC options
    parser.add_argument(
        "--noteevents-path",
        default="data/raw/NOTEEVENTS.csv",
        help="Path to MIMIC-III NOTEEVENTS.csv",
    )
    parser.add_argument(
        "--note-types",
        nargs="+",
        default=["Discharge summary"],
        help="MIMIC note types to process",
    )
    parser.add_argument(
        "--max-notes",
        type=int,
        default=None,
        help="Maximum MIMIC notes to process",
    )
    # Chunking options
    parser.add_argument(
        "--chunking",
        choices=["sentence", "semantic", "section"],
        default="sentence",
        help="Chunking strategy",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap in tokens",
    )
    # Output
    parser.add_argument(
        "--output",
        default="data/processed/",
        help="Output directory for processed documents",
    )
    return parser.parse_args()


def ingest_pubmed(args) -> Path:
    """Download and chunk PubMed abstracts."""
    output_dir = Path(args.output) / "pubmed"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = PubMedLoader()
    chunker = ChunkerFactory.create(
        strategy=args.chunking,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    all_docs = []

    if args.category:
        logger.info("Downloading category: %s", args.category)
        downloader = PubMedBatchDownloader(output_dir=output_dir, loader=loader)
        downloader.download_category(args.category, max_per_term=args.max_results)
        return output_dir

    mesh_terms = args.mesh_terms or []
    if args.query:
        records = loader.load_by_query(args.query, max_results=args.max_results)
        docs = [r.to_rag_document() for r in records]
        all_docs.extend(docs)

    for term in mesh_terms:
        logger.info("Downloading MeSH term: %s", term)
        records = loader.load_by_mesh_term(term, max_results=args.max_results)
        docs = [r.to_rag_document() for r in records]
        all_docs.extend(docs)

    if all_docs:
        # Chunk documents
        logger.info("Chunking %d documents...", len(all_docs))
        chunks = []
        for doc in all_docs:
            chunks.extend(chunker.chunk(doc))

        output_path = output_dir / "documents.jsonl"
        with open(output_path, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")

        logger.info(
            "Saved %d chunks from %d documents to %s",
            len(chunks),
            len(all_docs),
            output_path,
        )
        return output_path

    logger.warning("No documents ingested. Check your --mesh-terms or --query arguments.")
    return output_dir


def ingest_mimic(args) -> Path:
    """Process MIMIC-III clinical notes."""
    output_dir = Path(args.output) / "mimic"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = MIMICIIIClinicalNoteLoader(
        note_types=args.note_types,
        max_notes=args.max_notes,
        apply_extra_deidentification=True,
    )

    chunker = ChunkerFactory.create(
        strategy="section" if args.chunking == "section" else "sentence",
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    noteevents_path = Path(args.noteevents_path)
    logger.info("Processing MIMIC-III notes from %s", noteevents_path)

    output_path = output_dir / "clinical_notes.jsonl"
    chunk_count = 0

    with open(output_path, "w") as out_f:
        for note in loader.iter_notes(noteevents_path):
            for rag_doc in note.to_rag_documents(by_section=True):
                chunks = chunker.chunk(rag_doc)
                for chunk in chunks:
                    out_f.write(json.dumps(chunk.to_dict()) + "\n")
                    chunk_count += 1

    logger.info("Saved %d chunks to %s", chunk_count, output_path)
    logger.info("Processing stats: %s", loader.stats)
    return output_path


def main():
    args = parse_args()
    logger.info("ClinicalRAG Ingestion Pipeline")
    logger.info("Source: %s | Chunking: %s", args.source, args.chunking)

    if args.source in ("pubmed", "both"):
        pubmed_out = ingest_pubmed(args)
        logger.info("PubMed ingestion complete: %s", pubmed_out)

    if args.source in ("mimic", "both"):
        mimic_out = ingest_mimic(args)
        logger.info("MIMIC ingestion complete: %s", mimic_out)

    logger.info("Ingestion complete. Run build_index.py to create the vector index.")


if __name__ == "__main__":
    main()
