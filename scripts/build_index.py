#!/usr/bin/env python3
"""
Build vector index from processed documents.

Reads chunked JSONL documents, generates embeddings with the configured
biomedical embedding model, and builds a FAISS + BM25 hybrid index.

Usage:
    python scripts/build_index.py \\
        --input data/processed/pubmed/ \\
        --index-type hybrid \\
        --embedder pubmedbert \\
        --output data/indices/pubmed_ai_cds/

    # Build from config
    python scripts/build_index.py --config configs/pubmed_rag_config.yaml

    # Rebuild specific index type
    python scripts/build_index.py --input data/processed/ --index-type faiss_ivf \\
        --nlist 200 --output data/indices/faiss_ivf/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.biomedical_embedder import BiomedicalEmbedder
from src.vectorstore.vector_index import FAISSIndex, BM25Index, HybridIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ClinicalRAG vector index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, help="Input JSONL file or directory")
    parser.add_argument(
        "--output", default="data/indices/default", help="Output index directory"
    )
    parser.add_argument(
        "--index-type",
        choices=["hybrid", "faiss_flat", "faiss_ivf", "faiss_hnsw"],
        default="hybrid",
    )
    parser.add_argument(
        "--embedder",
        default="pubmedbert",
        help="Embedding model ID (see MODEL_REGISTRY)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Embedding batch size"
    )
    parser.add_argument("--nlist", type=int, default=100, help="IVF nlist parameter")
    parser.add_argument("--nprobe", type=int, default=10, help="IVF nprobe parameter")
    parser.add_argument("--config", help="Config file (overrides other args)")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to index (for testing)",
    )
    return parser.parse_args()


def load_documents(input_path: str, max_docs: int = None) -> List[dict]:
    """Load documents from JSONL file or directory of JSONL files."""
    input_path = Path(input_path)
    files = []

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL files found in {input_path}")

    documents = []
    for file in files:
        logger.info("Loading %s ...", file)
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    # Accept both chunk format and raw document format
                    if "chunk_id" in doc:
                        documents.append({
                            "id": doc["chunk_id"],
                            "text": doc["text"],
                            "metadata": doc.get("metadata", {}),
                        })
                    elif "id" in doc:
                        documents.append(doc)
                    if max_docs and len(documents) >= max_docs:
                        break
        if max_docs and len(documents) >= max_docs:
            break

    logger.info("Loaded %d documents from %d files", len(documents), len(files))
    return documents


def build_index(args):
    start = time.time()

    # Load documents
    documents = load_documents(args.input, max_docs=args.max_docs)
    if not documents:
        logger.error("No documents loaded. Check --input path.")
        sys.exit(1)

    texts = [doc.get("text", "") for doc in documents]

    # Initialize embedder
    logger.info("Initializing embedder: %s", args.embedder)
    embedder = BiomedicalEmbedder(
        model_id=args.embedder,
        batch_size=args.batch_size,
    )

    # Generate embeddings
    logger.info("Generating embeddings for %d documents...", len(documents))
    embed_start = time.time()
    embeddings = embedder.encode(texts, show_progress=True)
    embed_time = time.time() - embed_start
    logger.info(
        "Embeddings generated: shape=%s, time=%.1fs (%.0f docs/s)",
        embeddings.shape,
        embed_time,
        len(documents) / embed_time,
    )

    # Build index
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_type = args.index_type
    logger.info("Building %s index...", index_type)

    if index_type == "hybrid":
        faiss_type = "ivf" if len(documents) > 50000 else "flat"
        index = HybridIndex(
            dense_index=FAISSIndex(
                index_type=faiss_type,
                embedding_dim=embedder.dim,
                nlist=args.nlist,
                nprobe=args.nprobe,
            ),
            sparse_index=BM25Index(),
            embedding_dim=embedder.dim,
        )
        index.add_documents(documents, embeddings)
        index.save(output_dir)

    elif index_type.startswith("faiss"):
        faiss_variant = index_type.replace("faiss_", "")
        index = FAISSIndex(
            index_type=faiss_variant,
            embedding_dim=embedder.dim,
            nlist=args.nlist,
            nprobe=args.nprobe,
        )
        if faiss_variant == "ivf":
            index.train(embeddings)
        index.add_documents(documents, embeddings)
        index.save(output_dir)

    total_time = time.time() - start
    logger.info(
        "Index built: %d documents, dim=%d, time=%.1fs",
        len(documents),
        embedder.dim,
        total_time,
    )
    logger.info("Index saved to: %s", output_dir)
    logger.info("Run scripts/query.py --index %s to query.", output_dir)


def main():
    args = parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        # Override from config if not explicitly provided
        if not args.output:
            args.output = config.get("index", {}).get("path", args.output)
        if not args.embedder:
            args.embedder = config.get("embedding", {}).get("model_id", args.embedder)
        if not args.batch_size:
            args.batch_size = config.get("embedding", {}).get("batch_size", args.batch_size)

    build_index(args)


if __name__ == "__main__":
    main()
