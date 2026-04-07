"""
Setup configuration for ClinicalRAG package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme = Path("README.md").read_text(encoding="utf-8")

# Core dependencies
install_requires = [
    "torch>=2.0.0",
    "transformers>=4.38.0",
    "accelerate>=0.26.0",
    "sentence-transformers>=2.5.0",
    "datasets>=2.17.0",
    "faiss-cpu>=1.7.4",
    "requests>=2.31.0",
    "pyyaml>=6.0.1",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "tqdm>=4.66.0",
    "python-dotenv>=1.0.0",
]

# Optional extras
extras_require = {
    "openai": ["openai>=1.12.0"],
    "scispacy": ["scispacy>=0.5.3", "spacy>=3.7.0"],
    "eval": ["rouge-score>=0.1.2", "bert-score>=0.3.13", "evaluate>=0.4.1", "seqeval>=1.2.2"],
    "chromadb": ["chromadb>=0.4.22"],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=24.1.0",
        "isort>=5.13.0",
        "mypy>=1.7.0",
        "pre-commit>=3.5.0",
    ],
    "all": [
        "openai>=1.12.0",
        "scispacy>=0.5.3",
        "spacy>=3.7.0",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.13",
        "evaluate>=0.4.1",
        "seqeval>=1.2.2",
        "chromadb>=0.4.22",
        "rich>=13.7.0",
    ],
}

setup(
    name="clinical-nlp-rag",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description=(
        "Production-grade RAG pipeline for biomedical literature and clinical text "
        "with hallucination detection, biomedical NER, and relation extraction."
    ),
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/clinical-nlp-rag",
    packages=find_packages(exclude=["tests*", "scripts*", "docs*"]),
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=[
        "biomedical nlp",
        "clinical nlp",
        "rag",
        "retrieval augmented generation",
        "pubmed",
        "mimic",
        "named entity recognition",
        "relation extraction",
        "hallucination detection",
        "healthcare ai",
    ],
    project_urls={
        "Source": "https://github.com/yourusername/clinical-nlp-rag",
        "Bug Reports": "https://github.com/yourusername/clinical-nlp-rag/issues",
    },
    entry_points={
        "console_scripts": [
            "clinicalrag-ingest=scripts.ingest:main",
            "clinicalrag-build-index=scripts.build_index:main",
            "clinicalrag-query=scripts.query:main",
            "clinicalrag-evaluate=scripts.evaluate:main",
        ],
    },
)
