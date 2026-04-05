"""Central configuration for PubMed RAG."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "")  # optional — 10 req/s vs 3 req/s
NCBI_EMAIL: str = os.getenv("NCBI_EMAIL", "researcher@example.com")
API_KEY: str = os.getenv("API_KEY", "")  # optional — protects /ask and /ingest if set

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Local biomedical embedding model — trained on PubMed, no API key required.
# Quality is comparable to OpenAI embeddings for biomedical retrieval.
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"

CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "pubmed_retina"

# Default search queries for initial ingestion
DEFAULT_QUERIES = [
    "retinopathy of prematurity",
    "vaso-obliteration retina oxygen-induced",
    "neovascularization retina OIR",
    "VO zone NV zone retinal flatmount",
    "retinal vasculature development prematurity",
    "VEGF retinopathy prematurity treatment",
    "ROP anti-VEGF bevacizumab ranibizumab",
    "retinal avascular zone fluorescence angiography",
]

MAX_RESULTS_PER_QUERY = 200  # PubMed free limit per request
FULLTEXT_CHUNK_SIZE = 1000  # chars per chunk for PMC full-text sections

# RAG settings
TOP_K = 8  # number of abstracts to retrieve per query
MIN_SCORE = 0.35  # minimum cosine similarity — results below this are discarded
LLM_MODEL = "claude-haiku-4-5"  # fast + cheap; ideal for RAG Q&A
LLM_MAX_TOKENS = 2048

# Hybrid search (BM25 + vector)
HYBRID_SEARCH = True  # enable BM25 + vector fusion
RRF_K = 60  # Reciprocal Rank Fusion constant
RETRIEVAL_CANDIDATES = 20  # over-retrieve before fusion/re-ranking, then trim to TOP_K
