"""
PubMed data loader using NCBI Entrez API.

Supports batch downloading of abstracts by MeSH terms, PMIDs, or free-text
queries. Implements rate limiting, response caching, and structured XML
parsing into a normalized document schema.

NCBI Entrez API guidelines:
  - Maximum 3 requests/second without API key
  - Maximum 10 requests/second with API key (set NCBI_API_KEY env var)
  - Use history server (usehistory=y) for large result sets

References:
  Entrez Programming Utilities: https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Iterator, List, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PubMedAbstract:
    """Structured representation of a PubMed record."""

    pmid: str
    title: str
    abstract: str
    authors: List[str]
    mesh_terms: List[str]
    journal: str
    pub_date: str
    doi: Optional[str] = None
    pmc_id: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    publication_types: List[str] = field(default_factory=list)

    # Derived at load time
    source: str = "pubmed"
    retrieved_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_rag_document(self) -> dict:
        """
        Convert to the RAG document schema used throughout the pipeline.

        Returns:
            dict with keys: id, text, metadata
        """
        text = f"Title: {self.title}\n\nAbstract: {self.abstract}"
        metadata = {
            "pmid": self.pmid,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "pub_date": self.pub_date,
            "mesh_terms": self.mesh_terms,
            "keywords": self.keywords,
            "doi": self.doi,
            "source": self.source,
            "publication_types": self.publication_types,
        }
        return {"id": f"pmid_{self.pmid}", "text": text, "metadata": metadata}


# ---------------------------------------------------------------------------
# Entrez client
# ---------------------------------------------------------------------------

class EntrezClient:
    """
    Low-level NCBI Entrez API client with retry logic and rate limiting.

    Args:
        api_key: NCBI API key. If None, reads NCBI_API_KEY env var.
                 Without a key, rate limit is 3 req/s; with key, 10 req/s.
        email: Contact email required by NCBI usage policy.
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: str = "researcher@institution.edu",
    ):
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        self.email = email
        self._rate_limit = 0.1 if self.api_key else 0.34  # seconds between requests
        self._last_request_time: float = 0.0

        # Set up session with retry on transient failures
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        self._session = session

    def _wait_for_rate_limit(self) -> None:
        """Enforce NCBI rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

    def _get(self, endpoint: str, params: dict) -> str:
        """
        Make a GET request to the Entrez API.

        Args:
            endpoint: API endpoint (e.g., 'esearch', 'efetch').
            params: Query parameters.

        Returns:
            Response text.

        Raises:
            requests.HTTPError: On non-2xx responses after retries.
        """
        self._wait_for_rate_limit()
        url = f"{self.BASE_URL}/{endpoint}.fcgi"
        params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        response = self._session.get(url, params=params, timeout=30)
        self._last_request_time = time.time()
        response.raise_for_status()
        return response.text

    def esearch(
        self,
        query: str,
        db: str = "pubmed",
        retmax: int = 10000,
        use_history: bool = True,
    ) -> dict:
        """
        Search Entrez database for PMIDs matching a query.

        Args:
            query: Entrez query string (supports MeSH terms, Boolean operators).
            db: Entrez database (default: pubmed).
            retmax: Maximum number of IDs to return.
            use_history: If True, store results on NCBI history server for
                         efficient batched efetch.

        Returns:
            dict with keys: count, query_key, web_env, id_list
        """
        params = {
            "db": db,
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "usehistory": "y" if use_history else "n",
        }
        text = self._get("esearch", params)
        data = json.loads(text)["esearchresult"]

        logger.info(
            "ESearch '%s': found %s records (query_key=%s)",
            query,
            data.get("count"),
            data.get("querykey"),
        )
        return {
            "count": int(data.get("count", 0)),
            "query_key": data.get("querykey"),
            "web_env": data.get("webenv"),
            "id_list": data.get("idlist", []),
        }

    def efetch_xml(
        self,
        pmids: Optional[List[str]] = None,
        query_key: Optional[str] = None,
        web_env: Optional[str] = None,
        retstart: int = 0,
        retmax: int = 200,
        db: str = "pubmed",
    ) -> str:
        """
        Fetch full records in PubMed XML format.

        Supports two modes:
        - Direct: supply ``pmids`` list.
        - History server: supply ``query_key`` and ``web_env`` from esearch.

        Args:
            pmids: List of PubMed IDs to fetch.
            query_key: History server query key from esearch.
            web_env: History server web environment from esearch.
            retstart: Offset for pagination.
            retmax: Number of records to fetch (max 10000 per request).

        Returns:
            PubMed XML string.
        """
        params: dict = {
            "db": db,
            "rettype": "abstract",
            "retmode": "xml",
            "retstart": retstart,
            "retmax": retmax,
        }
        if pmids:
            params["id"] = ",".join(pmids)
        elif query_key and web_env:
            params["query_key"] = query_key
            params["WebEnv"] = web_env
        else:
            raise ValueError("Provide either pmids or (query_key + web_env).")

        return self._get("efetch", params)


# ---------------------------------------------------------------------------
# XML parser
# ---------------------------------------------------------------------------

class PubMedXMLParser:
    """Parse PubMed XML fetch results into PubMedAbstract objects."""

    def parse(self, xml_text: str) -> List[PubMedAbstract]:
        """
        Parse a PubMed XML response.

        Args:
            xml_text: Raw XML string from efetch.

        Returns:
            List of PubMedAbstract objects.
        """
        root = ET.fromstring(xml_text)
        records = []
        for article_node in root.findall(".//PubmedArticle"):
            try:
                record = self._parse_article(article_node)
                records.append(record)
            except Exception as exc:
                pmid = self._find_text(article_node, ".//PMID") or "unknown"
                logger.warning("Failed to parse PMID %s: %s", pmid, exc)
        return records

    def _parse_article(self, node: ET.Element) -> PubMedAbstract:
        pmid = self._find_text(node, ".//PMID") or ""
        title = self._find_text(node, ".//ArticleTitle") or ""
        abstract_text = self._parse_abstract(node)
        authors = self._parse_authors(node)
        mesh_terms = self._parse_mesh_terms(node)
        journal = self._find_text(node, ".//Journal/Title") or ""
        pub_date = self._parse_pub_date(node)
        doi = self._parse_doi(node)
        pmc_id = self._find_text(node, ".//ArticleIdList/ArticleId[@IdType='pmc']")
        keywords = [
            kw.text.strip()
            for kw in node.findall(".//KeywordList/Keyword")
            if kw.text
        ]
        pub_types = [
            pt.text.strip()
            for pt in node.findall(".//PublicationTypeList/PublicationType")
            if pt.text
        ]

        return PubMedAbstract(
            pmid=pmid,
            title=title,
            abstract=abstract_text,
            authors=authors,
            mesh_terms=mesh_terms,
            journal=journal,
            pub_date=pub_date,
            doi=doi,
            pmc_id=pmc_id,
            keywords=keywords,
            publication_types=pub_types,
        )

    def _find_text(self, node: ET.Element, xpath: str) -> Optional[str]:
        elem = node.find(xpath)
        if elem is not None and elem.text:
            return elem.text.strip()
        return None

    def _parse_abstract(self, node: ET.Element) -> str:
        """Handle structured abstracts (Background/Methods/Results/Conclusions)."""
        texts = []
        for abstract_text_node in node.findall(".//AbstractText"):
            label = abstract_text_node.get("Label")
            text = "".join(abstract_text_node.itertext()).strip()
            if label:
                texts.append(f"{label}: {text}")
            else:
                texts.append(text)
        return " ".join(texts)

    def _parse_authors(self, node: ET.Element) -> List[str]:
        authors = []
        for author in node.findall(".//AuthorList/Author"):
            last = self._find_text(author, "LastName") or ""
            fore = self._find_text(author, "ForeName") or ""
            if last:
                authors.append(f"{last}, {fore}".strip(", "))
        return authors

    def _parse_mesh_terms(self, node: ET.Element) -> List[str]:
        terms = []
        for mesh in node.findall(".//MeshHeadingList/MeshHeading"):
            descriptor = self._find_text(mesh, "DescriptorName")
            if descriptor:
                qualifiers = [
                    q.text.strip()
                    for q in mesh.findall("QualifierName")
                    if q.text
                ]
                if qualifiers:
                    terms.extend([f"{descriptor}/{q}" for q in qualifiers])
                else:
                    terms.append(descriptor)
        return terms

    def _parse_pub_date(self, node: ET.Element) -> str:
        year = self._find_text(node, ".//PubDate/Year")
        month = self._find_text(node, ".//PubDate/Month") or "01"
        day = self._find_text(node, ".//PubDate/Day") or "01"
        if year:
            return f"{year}-{month}-{day}"
        return ""

    def _parse_doi(self, node: ET.Element) -> Optional[str]:
        for article_id in node.findall(".//ArticleIdList/ArticleId"):
            if article_id.get("IdType") == "doi":
                return article_id.text.strip() if article_id.text else None
        return None


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------

class PubMedCache:
    """
    Disk-based cache for PubMed API responses.

    Stores raw XML responses as gzip-compressed files keyed by query hash.
    Avoids redundant API calls for identical queries.

    Args:
        cache_dir: Directory for cached response files.
        ttl_hours: Cache entry TTL in hours (default: 72).
    """

    def __init__(self, cache_dir: Path, ttl_hours: int = 72):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _key(self, data: str) -> str:
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """Return cached response for query, or None if missing/expired."""
        path = self.cache_dir / f"{self._key(query)}.xml.gz"
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.ttl_seconds:
            path.unlink()
            return None
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()

    def set(self, query: str, xml_text: str) -> None:
        """Cache a response."""
        path = self.cache_dir / f"{self._key(query)}.xml.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(xml_text)


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

class PubMedLoader:
    """
    High-level interface for loading PubMed abstracts.

    Combines Entrez API calls, XML parsing, rate limiting, and disk caching
    into a single easy-to-use interface.

    Args:
        api_key: NCBI API key (or set NCBI_API_KEY env var).
        email: Contact email for NCBI (required by usage policy).
        cache_dir: Directory for caching API responses.
        batch_size: Number of records per efetch request.

    Example::

        loader = PubMedLoader(email="me@hospital.org")

        # Fetch by MeSH term
        docs = loader.load_by_mesh_term("Atrial Fibrillation", max_results=1000)

        # Fetch by PMIDs
        docs = loader.load_by_pmids(["12345678", "87654321"])

        # Iterate without loading all into memory
        for doc in loader.iter_by_query("deep learning AND radiology"):
            process(doc)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: str = "researcher@institution.edu",
        cache_dir: Path = Path("data/cache/pubmed"),
        batch_size: int = 200,
    ):
        self.client = EntrezClient(api_key=api_key, email=email)
        self.parser = PubMedXMLParser()
        self.cache = PubMedCache(cache_dir)
        self.batch_size = batch_size

    def load_by_mesh_term(
        self,
        mesh_term: str,
        max_results: int = 1000,
        date_range: Optional[tuple[str, str]] = None,
    ) -> List[PubMedAbstract]:
        """
        Load abstracts for a MeSH term.

        Args:
            mesh_term: MeSH descriptor (e.g., "Atrial Fibrillation").
            max_results: Maximum records to retrieve.
            date_range: Optional (start_year, end_year) tuple.

        Returns:
            List of PubMedAbstract objects.
        """
        query = f'"{mesh_term}"[MeSH Terms]'
        if date_range:
            query += f" AND {date_range[0]}:{date_range[1]}[PDAT]"
        return self.load_by_query(query, max_results=max_results)

    def load_by_pmids(self, pmids: List[str]) -> List[PubMedAbstract]:
        """
        Load abstracts for a list of PMIDs.

        Args:
            pmids: List of PubMed IDs.

        Returns:
            List of PubMedAbstract objects.
        """
        all_records: List[PubMedAbstract] = []
        for i in range(0, len(pmids), self.batch_size):
            batch = pmids[i : i + self.batch_size]
            cache_key = f"pmids:{'_'.join(sorted(batch))}"
            xml_text = self.cache.get(cache_key)
            if xml_text is None:
                logger.info("Fetching batch of %d PMIDs from Entrez", len(batch))
                xml_text = self.client.efetch_xml(pmids=batch)
                self.cache.set(cache_key, xml_text)
            else:
                logger.info("Cache hit for %d PMIDs", len(batch))
            all_records.extend(self.parser.parse(xml_text))
        return all_records

    def load_by_query(
        self, query: str, max_results: int = 1000
    ) -> List[PubMedAbstract]:
        """
        Load abstracts by Entrez query string.

        Args:
            query: Entrez query (supports MeSH, Boolean, field tags).
            max_results: Maximum number of results to return.

        Returns:
            List of PubMedAbstract objects.
        """
        return list(self.iter_by_query(query, max_results=max_results))

    def iter_by_query(
        self, query: str, max_results: int = 10000
    ) -> Iterator[PubMedAbstract]:
        """
        Lazily iterate over PubMed records for a query.

        More memory-efficient than load_by_query for large result sets —
        yields records batch-by-batch without holding all in memory.

        Args:
            query: Entrez query string.
            max_results: Maximum number of records to yield.

        Yields:
            PubMedAbstract objects.
        """
        search = self.client.esearch(query, retmax=max_results)
        total = min(search["count"], max_results)
        logger.info("Fetching %d records for query: %s", total, query)

        for start in range(0, total, self.batch_size):
            cache_key = f"query:{query}:start={start}:max={self.batch_size}"
            xml_text = self.cache.get(cache_key)
            if xml_text is None:
                xml_text = self.client.efetch_xml(
                    query_key=search["query_key"],
                    web_env=search["web_env"],
                    retstart=start,
                    retmax=self.batch_size,
                )
                self.cache.set(cache_key, xml_text)
            records = self.parser.parse(xml_text)
            for rec in records:
                yield rec

    def save_jsonl(
        self,
        records: List[PubMedAbstract],
        output_path: Path,
        as_rag_documents: bool = True,
    ) -> None:
        """
        Save records to a JSONL file.

        Args:
            records: List of PubMedAbstract objects.
            output_path: Destination file path.
            as_rag_documents: If True, save in RAG document schema.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                doc = rec.to_rag_document() if as_rag_documents else rec.to_dict()
                f.write(json.dumps(doc) + "\n")
        logger.info("Saved %d records to %s", len(records), output_path)


# ---------------------------------------------------------------------------
# Batch downloader
# ---------------------------------------------------------------------------

class PubMedBatchDownloader:
    """
    Download large PubMed datasets by MeSH category.

    Supports downloading entire disease/drug/procedure categories for
    building comprehensive biomedical knowledge bases.

    Example::

        downloader = PubMedBatchDownloader(output_dir=Path("data/raw/pubmed"))
        downloader.download_category("Cardiovascular Diseases", max_per_term=2000)
    """

    # Curated MeSH term sets for common clinical domains
    CLINICAL_DOMAINS = {
        "cardiology": [
            "Atrial Fibrillation",
            "Heart Failure",
            "Myocardial Infarction",
            "Coronary Artery Disease",
            "Hypertension",
        ],
        "oncology": [
            "Breast Neoplasms",
            "Lung Neoplasms",
            "Colorectal Neoplasms",
            "Leukemia",
            "Lymphoma",
        ],
        "neurology": [
            "Alzheimer Disease",
            "Parkinson Disease",
            "Stroke",
            "Epilepsy",
            "Multiple Sclerosis",
        ],
        "pharmacology": [
            "Adverse Drug Reaction Reporting Systems",
            "Drug Interactions",
            "Drug Toxicity",
            "Pharmacokinetics",
        ],
        "ai_clinical": [
            "Artificial Intelligence",
            "Machine Learning",
            "Natural Language Processing",
            "Clinical Decision Support Systems",
            "Electronic Health Records",
        ],
    }

    def __init__(
        self,
        output_dir: Path = Path("data/processed/pubmed"),
        loader: Optional[PubMedLoader] = None,
    ):
        self.output_dir = Path(output_dir)
        self.loader = loader or PubMedLoader()

    def download_category(
        self,
        category: str,
        max_per_term: int = 1000,
        date_range: Optional[tuple[str, str]] = None,
    ) -> Path:
        """
        Download all MeSH terms for a clinical domain category.

        Args:
            category: One of the keys in CLINICAL_DOMAINS.
            max_per_term: Max records per MeSH term.
            date_range: Optional year range filter.

        Returns:
            Path to the output JSONL file.
        """
        terms = self.CLINICAL_DOMAINS.get(category)
        if not terms:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Available: {list(self.CLINICAL_DOMAINS.keys())}"
            )

        all_records: List[PubMedAbstract] = []
        seen_pmids: set[str] = set()

        for term in terms:
            logger.info("Downloading MeSH term: %s", term)
            records = self.loader.load_by_mesh_term(
                term, max_results=max_per_term, date_range=date_range
            )
            for rec in records:
                if rec.pmid not in seen_pmids:
                    all_records.append(rec)
                    seen_pmids.add(rec.pmid)

        output_path = self.output_dir / f"{category}.jsonl"
        self.loader.save_jsonl(all_records, output_path)
        logger.info(
            "Category '%s': %d unique records saved to %s",
            category,
            len(all_records),
            output_path,
        )
        return output_path
