"""Retrieval-augmented generation over the local PubMed ChromaDB collection.

Usage (interactive CLI):
    python rag.py "What causes vaso-obliteration in ROP?"
    python rag.py --search-only "VEGF retina"   # retrieval only, no LLM
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Iterator

import anthropic
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    ANTHROPIC_API_KEY,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    HYBRID_SEARCH,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    MIN_SCORE,
    RETRIEVAL_CANDIDATES,
    RRF_K,
    TOP_K,
)

# Module-level singletons — loaded once, reused across calls (e.g. in the API)
_embed_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_llm_client: anthropic.Anthropic | None = None
_load_lock = threading.Lock()

# BM25 index singletons
_bm25_index: BM25Okapi | None = None
_bm25_doc_ids: list[str] | None = None
_bm25_doc_count: int = 0  # track collection size for cache invalidation

SYSTEM_PROMPT = """\
You are a biomedical research assistant specialising in:
- Retinopathy of prematurity (ROP)
- Retinal vascular development and disease
- Vaso-obliteration (VO) and neovascularization (NV) zones
- Oxygen-induced retinopathy (OIR) animal models
- Anti-VEGF and other ROP treatments

Answer questions using ONLY the provided PubMed abstracts.
Cite each source with its reference number in square brackets, e.g. [1] or [1,3].
If the abstracts do not contain sufficient information, say so clearly.
Be precise and scientific in tone.\
"""

EXPLAIN_SYSTEM_PROMPT = """\
You are a biomedical image analysis assistant specialising in oxygen-induced retinopathy (OIR) \
and retinopathy of prematurity (ROP). You interpret quantitative segmentation results from \
retinal flatmount images.

Given segmentation percentages (NV = neovascular zone, VO = vaso-obliterated zone, \
Retina = non-vascular retinal background tissue) and experiment metadata parsed from the image filename, \
provide a concise scientific interpretation that covers:

1. **Disease stage assessment** — What do the NV% and VO% indicate about where this sample \
falls on the OIR timeline (vaso-obliteration phase vs. neovascular phase vs. regression)?
2. **Treatment group context** — If treatment information is present (e.g. PBS = vehicle \
control, a drug dose), how does it relate to the observed values?
3. **Comparison to expected ranges** — In the standard mouse OIR model, VO typically peaks \
around P12–P14 and NV peaks around P17. Comment on whether the values seem high, low, or \
typical given the postnatal day (if derivable from dates).
4. **Caveats** — Note any limitations (e.g. single eye, image quality issues inferred from \
context, uncertainty about treatment code meaning).

If PubMed abstracts are provided, cite supporting evidence with [reference numbers]. \
If no literature is available, rely on established OIR/ROP knowledge. \
Be concise (3–6 paragraphs). Do not speculate beyond what the data supports.\
"""


def _load_retriever() -> tuple[SentenceTransformer, chromadb.Collection]:
    """Load only the embedding model and vector store — no LLM client required."""
    global _embed_model, _collection
    with _load_lock:
        if _embed_model is None:
            _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        if _collection is None:
            client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            _collection = client.get_or_create_collection(COLLECTION_NAME)
    return _embed_model, _collection


def _load() -> tuple[SentenceTransformer, chromadb.Collection, anthropic.Anthropic]:
    """Load all components including the Anthropic LLM client."""
    global _llm_client
    model, collection = _load_retriever()
    with _load_lock:
        if _llm_client is None:
            if not ANTHROPIC_API_KEY:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Add it to your .env file or environment variables."
                )
            _llm_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return model, collection, _llm_client


# ---------------------------------------------------------------------------
#  BM25 index (lazy, rebuilt when collection grows)
# ---------------------------------------------------------------------------


def _build_bm25_index(collection: chromadb.Collection) -> None:
    """Build (or rebuild) the BM25 index from all ChromaDB documents."""
    global _bm25_index, _bm25_doc_ids, _bm25_doc_count

    data = collection.get(include=["documents"])
    doc_ids = data["ids"]
    documents = data["documents"]

    tokenized = [doc.lower().split() for doc in documents]
    _bm25_index = BM25Okapi(tokenized) if tokenized else None
    _bm25_doc_ids = doc_ids
    _bm25_doc_count = len(doc_ids)


def _ensure_bm25(collection: chromadb.Collection) -> BM25Okapi | None:
    """Return the BM25 index, rebuilding if the collection has grown."""
    if _bm25_index is None or collection.count() != _bm25_doc_count:
        _build_bm25_index(collection)
    return _bm25_index


def _bm25_search(
    query: str, collection: chromadb.Collection, top_k: int
) -> list[tuple[str, float]]:
    """Return top-k (doc_id, normalized_score) pairs via BM25."""
    bm25 = _ensure_bm25(collection)
    if bm25 is None or not _bm25_doc_ids:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
    if max_score <= 0:
        return []

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        (_bm25_doc_ids[idx], float(scores[idx]) / max_score)
        for idx in top_indices
        if scores[idx] > 0
    ]


def _rrf_merge(
    vector_results: list[dict],
    bm25_results: list[tuple[str, float]],
    k: int = RRF_K,
) -> list[dict]:
    """Merge vector and BM25 results via Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    # Vector results — use doc ID (pmid or chunk ID) for merging
    for rank, chunk in enumerate(vector_results):
        doc_id = chunk.get("pmid") or chunk.get("id", str(rank))
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        doc_map[doc_id] = chunk

    # BM25 results
    for rank, (doc_id, _bm25_score) in enumerate(bm25_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        # BM25 may return docs not in vector results — skip those
        # (we don't have their full metadata; they'll be picked up if they pass vector threshold)

    # Sort by RRF score, only return docs we have full metadata for
    ranked = sorted(
        [(doc_id, score) for doc_id, score in rrf_scores.items() if doc_id in doc_map],
        key=lambda x: x[1],
        reverse=True,
    )
    return [doc_map[doc_id] for doc_id, _ in ranked]


def retrieve(
    query: str,
    top_k: int = TOP_K,
    min_score: float = MIN_SCORE,
    hybrid: bool = HYBRID_SEARCH,
) -> list[dict]:
    """Return the top-k most relevant documents for a query."""
    model, collection = _load_retriever()

    if collection.count() == 0:
        return []

    n_candidates = max(top_k, RETRIEVAL_CANDIDATES) if hybrid else top_k

    embedding = model.encode([query], normalize_embeddings=True).tolist()[0]
    n_query = min(n_candidates, collection.count())
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_query,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(1.0 - float(dist) / 2, 4)  # cosine sim from L2² on unit-norm vecs
        if score < min_score:
            continue
        pmid = meta.get("pmid", "")
        pmcid = meta.get("pmcid", "")
        if pmcid:
            url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
        elif pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        else:
            url = ""
        chunks.append(
            {
                "text": doc,
                "pmid": pmid,
                "pmcid": pmcid,
                "title": meta["title"],
                "authors": meta.get("authors", ""),
                "year": meta.get("year", ""),
                "journal": meta.get("journal", ""),
                "section": meta.get("section", ""),
                "score": score,
                "url": url,
                "source": meta.get("source", "pubmed"),
                "filename": meta.get("filename", ""),
            }
        )

    if hybrid and chunks:
        bm25_results = _bm25_search(query, collection, n_candidates)
        if bm25_results:
            chunks = _rrf_merge(chunks, bm25_results)

    return chunks[:top_k]


def ask(question: str, top_k: int = TOP_K) -> dict:
    """Retrieve relevant abstracts and generate a grounded answer via Claude.

    Returns:
        {
            "answer": str,
            "sources": [{"ref", "pmid", "title", "authors", "year", "journal",
                         "score", "url"}, ...]
        }
    """
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        _, collection = _load_retriever()
        if collection.count() == 0:
            answer = "The database is empty. Run `python ingest.py` first to populate it."
        else:
            answer = (
                "No abstracts above the relevance threshold were found for this question. "
                "Try rephrasing, broadening your query, or lowering MIN_SCORE in config.py."
            )
        return {"answer": answer, "sources": []}

    # Build numbered context block
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] PMID {c['pmid']} ({c['year']}) — {c['title']}\n"
            f"Authors: {c['authors']} | Journal: {c['journal']}\n\n"
            f"{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"PubMed abstracts:\n\n{context}\n\nQuestion: {question}"

    _, _, llm = _load()
    response = llm.messages.create(
        model=LLM_MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=LLM_MAX_TOKENS,
    )

    answer = next((b.text for b in response.content if b.type == "text"), "")
    if response.stop_reason == "max_tokens":
        answer += "\n\n[Note: response truncated — increase LLM_MAX_TOKENS in config.py]"
    sources = [
        {
            "ref": i + 1,
            "pmid": c["pmid"],
            "pmcid": c.get("pmcid", ""),
            "title": c["title"],
            "authors": c["authors"],
            "year": c["year"],
            "journal": c["journal"],
            "section": c.get("section", ""),
            "score": c["score"],
            "url": c["url"],
        }
        for i, c in enumerate(chunks)
    ]

    return {"answer": answer, "sources": sources}


def ask_stream(question: str, top_k: int = TOP_K) -> tuple[list[dict], Iterator[str]]:
    """Retrieve abstracts (fast), then stream the LLM answer token-by-token.

    Returns:
        sources  — list of source dicts (available immediately, before LLM starts)
        stream   — generator yielding text chunks as they arrive
    """
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        _, collection = _load_retriever()
        if collection.count() == 0:
            msg = "The database is empty. Run `python ingest.py` first to populate it."
        else:
            msg = (
                "No abstracts above the relevance threshold were found. "
                "Try rephrasing or lowering MIN_SCORE in config.py."
            )

        def _empty() -> Iterator[str]:
            yield msg

        return [], _empty()

    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] PMID {c['pmid']} ({c['year']}) — {c['title']}\n"
            f"Authors: {c['authors']} | Journal: {c['journal']}\n\n"
            f"{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    user_message = f"PubMed abstracts:\n\n{context}\n\nQuestion: {question}"

    sources = [
        {
            "ref": i + 1,
            "pmid": c["pmid"],
            "pmcid": c.get("pmcid", ""),
            "title": c["title"],
            "authors": c["authors"],
            "year": c["year"],
            "journal": c["journal"],
            "section": c.get("section", ""),
            "score": c["score"],
            "url": c["url"],
            "filename": c.get("filename", ""),
        }
        for i, c in enumerate(chunks)
    ]

    _, _, llm = _load()

    def _stream() -> Iterator[str]:
        with llm.messages.stream(
            model=LLM_MODEL,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=LLM_MAX_TOKENS,
        ) as stream:
            yield from stream.text_stream

    return sources, _stream()


def chat_stream(question: str, history: list[dict]) -> tuple[list[dict], Iterator[str]]:
    """RAG with conversation history. Returns (sources, text_stream).

    Args:
        question: current user question
        history:  prior turns as [{"role": "user"|"assistant", "content": str}, ...]
    """
    chunks = retrieve(question)

    if not chunks:
        _, collection = _load_retriever()
        msg = (
            "The database is empty. Run `python ingest.py` first."
            if collection.count() == 0
            else "No abstracts found above the relevance threshold. Try rephrasing."
        )

        def _empty() -> Iterator[str]:
            yield msg

        return [], _empty()

    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] PMID {c['pmid']} ({c['year']}) — {c['title']}\n"
            f"Authors: {c['authors']} | Journal: {c['journal']}\n\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    user_message = f"PubMed abstracts:\n\n{context}\n\nQuestion: {question}"

    sources = [
        {
            "ref": i + 1,
            "pmid": c["pmid"],
            "pmcid": c.get("pmcid", ""),
            "title": c["title"],
            "authors": c["authors"],
            "year": c["year"],
            "journal": c["journal"],
            "section": c.get("section", ""),
            "score": c["score"],
            "url": c["url"],
            "filename": c.get("filename", ""),
        }
        for i, c in enumerate(chunks)
    ]

    api_messages = list(history) + [{"role": "user", "content": user_message}]

    _, _, llm = _load()

    def _stream() -> Iterator[str]:
        with llm.messages.stream(
            model=LLM_MODEL,
            system=SYSTEM_PROMPT,
            messages=api_messages,
            max_tokens=LLM_MAX_TOKENS,
        ) as stream:
            yield from stream.text_stream

    return sources, _stream()


def explain_segmentation(
    metrics: dict,
    metadata: dict,
    top_k: int = TOP_K,
) -> tuple[list[dict], Iterator[str]]:
    """Generate a RAG-grounded biological interpretation of segmentation results.

    Args:
        metrics:  {"nv_pct": float, "vo_pct": float, "retina_pct": float, ...}
        metadata: parsed filename fields — any subset of:
                  raw, birth_date, image_date, postnatal_day,
                  experiment, protocol, treatment, animal_info, eye, litter
        top_k:    number of PubMed abstracts to retrieve

    Returns:
        (sources, text_stream)
    """
    # Build a targeted retrieval query from experiment context
    query_parts = ["OIR neovascularization vaso-obliteration retinal flatmount segmentation"]
    if metadata.get("experiment"):
        query_parts.append(metadata["experiment"])
    if metadata.get("treatment") and metadata["treatment"].lower() not in ("pbs", ""):
        query_parts.append(f"treatment {metadata['treatment']}")
    chunks = retrieve(" ".join(query_parts), top_k=top_k)

    # Build the experiment context block
    meta_lines: list[str] = []
    if metadata.get("experiment"):
        meta_lines.append(f"Experiment type: {metadata['experiment']}")
    if metadata.get("birth_date"):
        meta_lines.append(f"Birth date: {metadata['birth_date']}")
    if metadata.get("image_date"):
        meta_lines.append(f"Image / sacrifice date: {metadata['image_date']}")
    if metadata.get("postnatal_day") is not None:
        meta_lines.append(f"Postnatal day at imaging (P): {metadata['postnatal_day']}")
    if metadata.get("protocol"):
        meta_lines.append(f"Protocol / annotation: {metadata['protocol']}")
    if metadata.get("litter"):
        meta_lines.append(f"Litter: {metadata['litter']}")
    if metadata.get("eye"):
        meta_lines.append(f"Eye: {metadata['eye']}")
    if metadata.get("treatment"):
        meta_lines.append(f"Treatment group: {metadata['treatment']}")
    if metadata.get("animal_info"):
        meta_lines.append(f"Animal identifier: {metadata['animal_info']}")
    meta_lines.append(f"Image filename: {metadata.get('raw', 'unknown')}")

    # Format metrics
    metrics_lines = []
    for key, label in [
        ("nv", "Neovascular (NV)"),
        ("vo", "Vaso-obliterated (VO)"),
        ("retina", "Retina (non-vascular retinal tissue)"),
    ]:
        pct = metrics.get(f"{key}_pct")
        if pct is not None:
            metrics_lines.append(f"  {label}: {pct:.2f}% of retinal area")

    # Assemble literature context
    if chunks:
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] PMID {c['pmid']} ({c['year']}) — {c['title']}\n"
                f"Authors: {c['authors']} | Journal: {c['journal']}\n\n{c['text']}"
            )
        lit_section = "\n\nRelevant PubMed abstracts:\n\n" + "\n\n---\n\n".join(context_parts)
    else:
        lit_section = "\n\n(No PubMed abstracts were retrieved for this query.)"

    user_message = (
        "Experiment metadata:\n"
        + "\n".join(meta_lines)
        + "\n\nSegmentation results:\n"
        + "\n".join(metrics_lines)
        + lit_section
        + "\n\nPlease interpret these segmentation results."
    )

    sources = [
        {
            "ref": i + 1,
            "pmid": c["pmid"],
            "pmcid": c.get("pmcid", ""),
            "title": c["title"],
            "authors": c["authors"],
            "year": c["year"],
            "journal": c["journal"],
            "section": c.get("section", ""),
            "score": c["score"],
            "url": c["url"],
            "filename": c.get("filename", ""),
        }
        for i, c in enumerate(chunks)
    ]

    _, _, llm = _load()

    def _stream() -> Iterator[str]:
        with llm.messages.stream(
            model=LLM_MODEL,
            system=EXPLAIN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=LLM_MAX_TOKENS,
        ) as stream:
            yield from stream.text_stream

    return sources, _stream()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question against the local PubMed RAG database."
    )
    parser.add_argument("question", help="Research question to answer")
    parser.add_argument(
        "--top-k", type=int, default=TOP_K, help=f"Abstracts to retrieve (default: {TOP_K})"
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Return retrieved abstracts only — skip LLM generation",
    )
    args = parser.parse_args()

    if args.search_only:
        chunks = retrieve(args.question, top_k=args.top_k)
        if not chunks:
            print("No results found. Run `python ingest.py` first.")
            return
        print(f"\nTop {len(chunks)} results for: '{args.question}'\n")
        for c in chunks:
            print(
                f"  [{c['score']:.3f}] PMID {c['pmid']} ({c['year']}) — {c['title']}\n"
                f"           {c['url']}\n"
            )
    else:
        result = ask(args.question, top_k=args.top_k)
        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(result["answer"])
        print("\n" + "=" * 70)
        print("SOURCES")
        print("=" * 70)
        for s in result["sources"]:
            print(
                f"  [{s['ref']}] {s['authors']} ({s['year']}). {s['title']}.\n"
                f"       {s['journal']} | {s['url']} | score: {s['score']}\n"
            )


if __name__ == "__main__":
    main()
