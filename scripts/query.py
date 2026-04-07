#!/usr/bin/env python3
"""
Interactive query script for ClinicalRAG.

Provides a command-line interface for querying the RAG system with:
  - Interactive REPL mode
  - Single-query mode
  - Literature review mode
  - Adverse event extraction mode

Usage:
    # Interactive REPL
    python scripts/query.py --index data/indices/pubmed_ai_cds/ --mode interactive

    # Single query
    python scripts/query.py --index data/indices/pubmed_ai_cds/ \\
        --query "What are the contraindications for warfarin?"

    # Literature review
    python scripts/query.py --index data/indices/pubmed_ai_cds/ \\
        --mode literature-review \\
        --query "Efficacy of SGLT2 inhibitors in heart failure"

    # Show retrieval details
    python scripts/query.py --index data/indices/ --query "..." --verbose
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)  # Quiet for interactive use
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ClinicalRAG query interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--index", required=True, help="Path to vector index directory"
    )
    parser.add_argument("--query", "-q", help="Query string (single-query mode)")
    parser.add_argument(
        "--mode",
        choices=["interactive", "single", "literature-review", "adverse-events"],
        default="single" if "--query" in sys.argv else "interactive",
    )
    parser.add_argument(
        "--config",
        default="configs/pubmed_rag_config.yaml",
        help="Config file",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument(
        "--embedder", default="pubmedbert", help="Embedding model"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--output",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder re-ranking (faster)",
    )
    parser.add_argument(
        "--hallucination-check",
        action="store_true",
        help="Run hallucination detection on generated answers",
    )
    return parser.parse_args()


def build_pipeline(args):
    """Build retrieval + generation pipeline from args."""
    from src.embeddings.biomedical_embedder import BiomedicalEmbedder
    from src.vectorstore.vector_index import HybridIndex
    from src.retrieval.retriever import (
        HybridRetriever,
        CrossEncoderReranker,
        UMLSQueryExpander,
    )
    from src.generation.rag_generator import ClinicalRAGGenerator, OpenAIProvider

    print(f"Loading index from {args.index}...")
    embedder = BiomedicalEmbedder(model_id=args.embedder)
    index = HybridIndex(embedding_dim=embedder.dim)
    index.load(Path(args.index))

    reranker = None
    if not args.no_rerank:
        reranker = CrossEncoderReranker()

    retriever = HybridRetriever(
        index=index,
        embedder=embedder,
        reranker=reranker,
        query_expander=UMLSQueryExpander(use_fallback=True),
    )

    generator = ClinicalRAGGenerator(
        provider=OpenAIProvider(model=args.model),
        include_disclaimer=True,
    )

    return retriever, generator


def run_query(
    query: str,
    retriever,
    generator,
    task: str = "clinical_qa",
    top_k: int = 5,
    verbose: bool = False,
    hallucination_check: bool = False,
):
    """Execute a single query and return results."""
    start = time.time()
    docs = retriever.retrieve(query, top_k=top_k)
    latency = (time.time() - start) * 1000

    answer = generator.generate(query, docs, task=task)

    if hallucination_check:
        from src.generation.hallucination_detector import HallucinationDetector
        detector = HallucinationDetector()
        report = detector.detect(answer.text, docs)
        answer.faithfulness_score = report.faithfulness_score
    
    return answer, docs, latency


def print_result(
    query: str,
    answer,
    docs,
    latency: float,
    verbose: bool = False,
):
    """Print formatted query result."""
    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    if verbose:
        print(f"\nRetrieved {len(docs)} documents (latency: {latency:.0f}ms):")
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            src = meta.get("pmid") or meta.get("note_id") or doc.chunk_id
            print(f"  [{i}] Score={doc.score:.3f} | {src}")
            print(f"       {doc.text[:120]}...")
        print()

    print("\nAnswer:")
    print("-" * 40)
    print(answer.text)

    if answer.citations:
        print("\nSources:")
        for i, cite in enumerate(answer.citations, 1):
            print(f"  [{i}] {cite.formatted}")

    if answer.faithfulness_score > 0:
        print(f"\nFaithfulness: {answer.faithfulness_score:.3f} | "
              f"Confidence: {answer.confidence:.2f}")

    print(f"\n[{answer.model} | {answer.generation_time_ms:.0f}ms generation]")


def interactive_mode(args, retriever, generator):
    """Run interactive REPL."""
    print("\nClinicalRAG Interactive Query Mode")
    print("Commands: 'exit' to quit, 'verbose' to toggle verbose, 'help' for help")
    print("-" * 60)

    verbose = args.verbose
    results = []

    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            break
        if query.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose: {'ON' if verbose else 'OFF'}")
            continue
        if query.lower() == "help":
            print(
                "Commands:\n"
                "  exit / quit   — Exit\n"
                "  verbose       — Toggle verbose mode\n"
                "  /lit <query>  — Literature review mode\n"
                "  /adr <query>  — Adverse event extraction\n"
            )
            continue

        # Detect task modifier
        task = "clinical_qa"
        if query.startswith("/lit "):
            task = "literature_synthesis"
            query = query[5:]
        elif query.startswith("/adr "):
            task = "adverse_event_extraction"
            query = query[5:]

        try:
            answer, docs, latency = run_query(
                query=query,
                retriever=retriever,
                generator=generator,
                task=task,
                top_k=args.top_k,
                verbose=verbose,
                hallucination_check=args.hallucination_check,
            )
            print_result(query, answer, docs, latency, verbose)
            results.append({"query": query, "answer": answer.text, "task": task})

        except Exception as exc:
            print(f"Error: {exc}")

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


def literature_review(
    question: str,
    mesh_terms: list = None,
    date_range: tuple = None,
    max_papers: int = 50,
    index_path: str = None,
    embedder_id: str = "pubmedbert",
) -> dict:
    """
    Generate a literature review for a clinical question.

    Used as a programmatic API for literature review automation.
    Retrieves relevant papers and synthesizes evidence across them.

    Args:
        question: Clinical research question.
        mesh_terms: Optional MeSH terms for focused retrieval.
        date_range: Optional (start_year, end_year) filter.
        max_papers: Maximum papers to include.
        index_path: Vector index path.
        embedder_id: Embedding model to use.

    Returns:
        Dict with 'synthesis', 'papers', and 'evidence_quality'.
    """
    from src.embeddings.biomedical_embedder import BiomedicalEmbedder
    from src.vectorstore.vector_index import HybridIndex
    from src.retrieval.retriever import HybridRetriever
    from src.generation.rag_generator import ClinicalRAGGenerator, OpenAIProvider

    embedder = BiomedicalEmbedder(model_id=embedder_id)
    index = HybridIndex(embedding_dim=embedder.dim)
    index.load(Path(index_path))
    retriever = HybridRetriever(index=index, embedder=embedder)
    generator = ClinicalRAGGenerator(provider=OpenAIProvider())

    # Add MeSH terms to query for specificity
    full_query = question
    if mesh_terms:
        full_query += " " + " ".join(mesh_terms)

    docs = retriever.retrieve(
        full_query,
        top_k=min(max_papers, 20),
        filter_metadata={} if not date_range else {"pub_date_year_gte": date_range[0]},
    )
    answer = generator.generate(question, docs, task="literature_synthesis")

    return {
        "synthesis": answer.text,
        "papers": [
            {
                "pmid": d.metadata.get("pmid"),
                "title": d.metadata.get("title"),
                "score": d.score,
            }
            for d in docs
        ],
        "num_papers": len(docs),
    }


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    retriever, generator = build_pipeline(args)

    if args.mode == "interactive" or (args.mode == "single" and not args.query):
        interactive_mode(args, retriever, generator)
    else:
        query = args.query
        task_map = {
            "literature-review": "literature_synthesis",
            "adverse-events": "adverse_event_extraction",
            "single": "clinical_qa",
        }
        task = task_map.get(args.mode, "clinical_qa")

        answer, docs, latency = run_query(
            query=query,
            retriever=retriever,
            generator=generator,
            task=task,
            top_k=args.top_k,
            verbose=args.verbose,
            hallucination_check=args.hallucination_check,
        )
        print_result(query, answer, docs, latency, args.verbose)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(
                    {
                        "query": query,
                        "answer": answer.text,
                        "citations": [c.to_dict() if hasattr(c, 'to_dict') else vars(c) for c in answer.citations],
                        "faithfulness": answer.faithfulness_score,
                    },
                    f,
                    indent=2,
                )


if __name__ == "__main__":
    main()
