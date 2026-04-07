#!/usr/bin/env python3
"""
Evaluation script for ClinicalRAG pipeline.

Runs standardized evaluation on configured benchmarks and produces
a comprehensive report with retrieval, generation, and faithfulness metrics.

Usage:
    # Run full evaluation suite
    python scripts/evaluate.py --config configs/pubmed_rag_config.yaml \\
        --index data/indices/pubmed_ai_cds/ --output results/

    # Single benchmark
    python scripts/evaluate.py --benchmark bioasq \\
        --data-path data/benchmarks/bioasq/ --index data/indices/pubmed_ai_cds/

    # Retrieval-only evaluation (no LLM needed)
    python scripts/evaluate.py --retrieval-only \\
        --qa-dataset data/eval/qa_with_doc_ids.jsonl \\
        --index data/indices/pubmed_ai_cds/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ClinicalRAG evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="configs/pubmed_rag_config.yaml", help="Config file"
    )
    parser.add_argument("--index", required=True, help="Vector index directory")
    parser.add_argument(
        "--benchmark",
        choices=["bioasq", "pubmedqa", "medqa", "i2b2", "all"],
        help="Specific benchmark to run",
    )
    parser.add_argument(
        "--qa-dataset",
        help="Path to custom QA dataset JSONL for RAGAS evaluation",
    )
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument(
        "--embedder", default="pubmedbert", help="Embedding model"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum examples per benchmark",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only evaluate retrieval (skip generation)",
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Skip BERTScore computation",
    )
    return parser.parse_args()


def build_pipeline(args):
    from src.embeddings.biomedical_embedder import BiomedicalEmbedder
    from src.vectorstore.vector_index import HybridIndex
    from src.retrieval.retriever import HybridRetriever, CrossEncoderReranker
    from src.generation.rag_generator import ClinicalRAGGenerator, OpenAIProvider
    from src.generation.hallucination_detector import HallucinationDetector

    embedder = BiomedicalEmbedder(model_id=args.embedder)
    index = HybridIndex(embedding_dim=embedder.dim)
    index.load(Path(args.index))

    retriever = HybridRetriever(
        index=index,
        embedder=embedder,
        reranker=CrossEncoderReranker() if not args.retrieval_only else None,
    )

    generator = None
    detector = None
    if not args.retrieval_only:
        generator = ClinicalRAGGenerator(
            provider=OpenAIProvider(model=args.model)
        )
        detector = HallucinationDetector()

    return retriever, generator, detector


def run_ragas_evaluation(args, retriever, generator):
    """Run RAGAS-style evaluation on custom QA dataset."""
    from src.evaluation.rag_evaluator import RAGEvaluator, RAGASEvaluator

    qa_dataset = []
    with open(args.qa_dataset) as f:
        for line in f:
            line = line.strip()
            if line:
                qa_dataset.append(json.loads(line))

    if args.max_examples:
        qa_dataset = qa_dataset[: args.max_examples]

    evaluator = RAGEvaluator(
        retriever=retriever,
        generator=generator,
        compute_bertscore=not args.no_bertscore,
    )

    report = evaluator.evaluate(
        qa_dataset=qa_dataset,
        save_path=Path(args.output) / "ragas_eval.json",
    )
    return report


def run_bioasq(args, retriever, generator):
    from src.evaluation.clinical_benchmarks import BioASQEvaluator

    data_path = Path(args.qa_dataset or "data/benchmarks/bioasq/")
    evaluator = BioASQEvaluator(retriever=retriever, generator=generator)
    result = evaluator.evaluate(data_path, max_examples=args.max_examples)
    return result


def run_pubmedqa(args, retriever, generator):
    from src.evaluation.clinical_benchmarks import PubMedQAEvaluator

    evaluator = PubMedQAEvaluator(retriever=retriever, generator=generator)
    result = evaluator.evaluate(
        use_hf_dataset=True,
        max_examples=args.max_examples,
    )
    return result


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building pipeline...")
    retriever, generator, detector = build_pipeline(args)

    results = {"timestamp": datetime.utcnow().isoformat(), "benchmarks": {}}

    benchmark = args.benchmark or ("all" if not args.qa_dataset else "custom")

    if args.qa_dataset:
        logger.info("Running RAGAS evaluation on %s", args.qa_dataset)
        try:
            report = run_ragas_evaluation(args, retriever, generator)
            report.print_summary()
            results["benchmarks"]["ragas"] = report.to_dict()
        except Exception as exc:
            logger.error("RAGAS evaluation failed: %s", exc)

    if benchmark in ("bioasq", "all"):
        logger.info("Running BioASQ evaluation...")
        try:
            r = run_bioasq(args, retriever, generator)
            print(r.summary())
            results["benchmarks"]["bioasq"] = r.to_dict()
        except Exception as exc:
            logger.warning("BioASQ skipped: %s", exc)

    if benchmark in ("pubmedqa", "all"):
        logger.info("Running PubMedQA evaluation...")
        try:
            r = run_pubmedqa(args, retriever, generator)
            print(f"PubMedQA accuracy: {r.accuracy:.4f} (n={r.num_examples})")
            results["benchmarks"]["pubmedqa"] = r.to_dict()
        except Exception as exc:
            logger.warning("PubMedQA skipped: %s", exc)

    # Save combined results
    report_path = output_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Evaluation report saved to %s", report_path)


if __name__ == "__main__":
    main()
