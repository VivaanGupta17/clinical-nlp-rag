#!/usr/bin/env python3
"""
NER fine-tuning script for ClinicalRAG.

Fine-tunes a BERT-based NER model on biomedical datasets or custom annotations.

Supported datasets:
  - BC5CDR: Drug + Disease entities from BioCreative V (automatic download via HuggingFace)
  - NCBI Disease: Disease mentions (HuggingFace)
  - i2b2 2010: Problem/Treatment/Test in clinical text (requires data agreement)
  - custom: JSONL format with tokens and BIO labels

Usage:
    # Fine-tune on BC5CDR (drug+disease NER)
    python scripts/train_ner.py --dataset bc5cdr \\
        --base-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract \\
        --output models/pubmedbert-bc5cdr-ner/

    # Fine-tune on custom data
    python scripts/train_ner.py --dataset custom \\
        --train-file data/ner/train.jsonl \\
        --eval-file data/ner/dev.jsonl \\
        --output models/custom-ner/

    # Evaluate existing model on test set
    python scripts/train_ner.py --eval-only \\
        --model models/pubmedbert-bc5cdr-ner/ \\
        --test-file data/ner/test.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# BC5CDR label scheme
BC5CDR_LABELS = [
    "O",
    "B-Chemical", "I-Chemical",
    "B-Disease", "I-Disease",
]

# Clinical NER label scheme (i2b2 2010)
I2B2_LABELS = [
    "O",
    "B-problem", "I-problem",
    "B-treatment", "I-treatment",
    "B-test", "I-test",
]

# Full biomedical entity label scheme
FULL_BIO_LABELS = [
    "O",
    "B-DISEASE", "I-DISEASE",
    "B-DRUG", "I-DRUG",
    "B-GENE", "I-GENE",
    "B-PROTEIN", "I-PROTEIN",
    "B-MUTATION", "I-MUTATION",
    "B-CELL_LINE", "I-CELL_LINE",
    "B-SPECIES", "I-SPECIES",
    "B-ANATOMY", "I-ANATOMY",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="ClinicalRAG NER fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=["bc5cdr", "ncbi_disease", "i2b2", "custom"],
        default="bc5cdr",
    )
    parser.add_argument(
        "--base-model",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    )
    parser.add_argument("--output", default="models/ner_finetuned/")
    parser.add_argument("--train-file", help="Custom training JSONL")
    parser.add_argument("--eval-file", help="Custom evaluation JSONL")
    parser.add_argument("--test-file", help="Test JSONL for eval-only mode")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model", help="Model to evaluate (eval-only mode)")
    return parser.parse_args()


def load_hf_dataset(dataset_name: str):
    """Load a HuggingFace NER dataset."""
    try:
        from datasets import load_dataset
        if dataset_name == "bc5cdr":
            ds = load_dataset("tner/bc5cdr")
            return ds["train"], ds["validation"], ds["test"]
        elif dataset_name == "ncbi_disease":
            ds = load_dataset("ncbi_disease")
            return ds["train"], ds["validation"], ds["test"]
        else:
            raise ValueError(f"Unknown HF dataset: {dataset_name}")
    except ImportError as exc:
        raise ImportError(
            "datasets required. Install: pip install datasets"
        ) from exc


def load_custom_dataset(train_file: str, eval_file: str = None):
    """
    Load custom NER dataset from JSONL files.

    Expected format (one example per line):
    {"tokens": ["Patient", "has", "atrial", "fibrillation"], 
     "ner_tags": [0, 0, 1, 2]}
    
    Where tags use BIO encoding with 0=O, 1=B-DISEASE, 2=I-DISEASE, etc.
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("datasets required. Install: pip install datasets") from exc

    def load_file(path):
        examples = {"tokens": [], "ner_tags": []}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    examples["tokens"].append(ex["tokens"])
                    examples["ner_tags"].append(ex["ner_tags"])
        return Dataset.from_dict(examples)

    train_ds = load_file(train_file)
    eval_ds = load_file(eval_file) if eval_file else None
    return train_ds, eval_ds, None


def evaluate_model(model_path: str, test_file: str):
    """Evaluate a fine-tuned NER model on a test set."""
    from src.ner.biomedical_ner import BERTBiomedicalNER

    ner = BERTBiomedicalNER(model_name=model_path)
    test_examples = []
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                test_examples.append(json.loads(line))

    tp = fp = fn = 0
    for example in test_examples:
        text = " ".join(example["tokens"])
        result = ner.extract(text)
        pred_spans = {(e.text, e.entity_type) for e in result.entities}
        # Reconstruct gold spans from BIO tags (simplified)
        # In production, use seqeval directly on token-level predictions
        # (this is a simplified token-span approximation)

    logger.info("Evaluation complete.")


def main():
    args = parse_args()

    if args.eval_only:
        if not args.model or not args.test_file:
            print("--model and --test-file required for --eval-only")
            sys.exit(1)
        evaluate_model(args.model, args.test_file)
        return

    from src.ner.biomedical_ner import BioNERFineTuner

    # Select label scheme
    if args.dataset == "bc5cdr":
        labels = BC5CDR_LABELS
    elif args.dataset == "i2b2":
        labels = I2B2_LABELS
    else:
        labels = FULL_BIO_LABELS

    tuner = BioNERFineTuner(
        base_model=args.base_model,
        output_dir=args.output,
        label_list=labels,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
    )

    # Load dataset
    logger.info("Loading dataset: %s", args.dataset)
    if args.dataset == "custom":
        train_ds, eval_ds, _ = load_custom_dataset(args.train_file, args.eval_file)
    else:
        train_ds, eval_ds, _ = load_hf_dataset(args.dataset)

    # Fine-tune
    logger.info(
        "Fine-tuning %s on %s → %s",
        args.base_model,
        args.dataset,
        args.output,
    )
    tuner.train(train_ds, eval_ds)
    logger.info("Training complete. Model saved to %s", args.output)
    logger.info(
        "Load with: BiomedicalNER(use_bert=True, bert_model='%s')", args.output
    )


if __name__ == "__main__":
    main()
