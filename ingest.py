"""Fetch PubMed abstracts and embed them into a local ChromaDB collection.

Usage:
    python ingest.py                              # default retina/ROP queries
    python ingest.py "vaso-obliteration" "VEGF"   # custom queries
    python ingest.py --max 500                    # fetch more per query
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import chromadb
from Bio import Entrez
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DEFAULT_QUERIES,
    EMBEDDING_MODEL,
    MAX_RESULTS_PER_QUERY,
    NCBI_API_KEY,
    NCBI_EMAIL,
)


def _setup_entrez() -> None:
    Entrez.email = NCBI_EMAIL
    if NCBI_API_KEY:
        Entrez.api_key = NCBI_API_KEY


def fetch_pmids(query: str, max_results: int) -> list[str]:
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return list(record["IdList"])


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch title + abstract XML for a list of PMIDs. Returns parsed dicts."""
    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="xml",
    )
    records = Entrez.read(handle)
    handle.close()

    articles = []
    for record in records["PubmedArticle"]:
        try:
            medline = record["MedlineCitation"]
            article = medline["Article"]

            title = str(article.get("ArticleTitle", ""))

            abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_parts, list):
                abstract = " ".join(str(p) for p in abstract_parts)
            else:
                abstract = str(abstract_parts)

            if not abstract.strip():
                continue  # skip articles with no abstract

            pmid = str(medline["PMID"])

            author_list = article.get("AuthorList", [])
            authors_parsed = []
            for auth in author_list:
                last = auth.get("LastName", "")
                fore = auth.get("ForeName", "")
                if last:
                    authors_parsed.append(f"{last} {fore}".strip())
            authors = ", ".join(authors_parsed[:3]) + (" et al." if len(authors_parsed) > 3 else "")

            pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = str(pub_date.get("Year", str(pub_date.get("MedlineDate", ""))[:4]))
            journal = str(article.get("Journal", {}).get("Title", ""))

            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": year,
                    "journal": journal,
                    # Full text passed to the embedding model
                    "text": f"Title: {title}\n\nAbstract: {abstract}",
                }
            )
        except (KeyError, IndexError):
            continue

    return articles


def ingest(
    queries: list[str],
    max_per_query: int = MAX_RESULTS_PER_QUERY,
    model: SentenceTransformer | None = None,
) -> int:
    """Fetch papers for all queries, embed, and store in ChromaDB.

    Returns the number of newly added documents.
    Pass a pre-loaded SentenceTransformer to avoid reloading the model from disk.
    """
    _setup_entrez()

    if model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
    else:
        print("Reusing pre-loaded embedding model.")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[])["ids"])
    print(f"Documents already in DB: {len(existing_ids)}")

    # Collect all unique articles across queries
    all_articles: list[dict] = []
    seen_pmids: set[str] = set(existing_ids)

    for query in queries:
        print(f"\nSearching PubMed: '{query}'")
        try:
            pmids = fetch_pmids(query, max_per_query)
        except Exception as exc:
            print(f"  Warning: failed to fetch PMIDs — {exc}")
            continue
        new_pmids = [p for p in pmids if p not in seen_pmids]
        print(f"  {len(pmids)} results, {len(new_pmids)} new")
        seen_pmids.update(new_pmids)

        # Fetch in batches of 100 (NCBI limit)
        for i in range(0, len(new_pmids), 100):
            batch_pmids = new_pmids[i : i + 100]
            try:
                articles = fetch_abstracts(batch_pmids)
            except Exception as exc:
                print(f"  Warning: failed to fetch abstracts for batch — {exc}")
                continue
            all_articles.extend(articles)
            time.sleep(0.35 if not NCBI_API_KEY else 0.11)  # respect rate limits

    if not all_articles:
        print("\nNothing new to add.")
        return 0

    print(f"\nEmbedding {len(all_articles)} new articles …")

    batch_size = 64
    for i in tqdm(range(0, len(all_articles), batch_size), desc="Embedding"):
        batch = all_articles[i : i + batch_size]
        texts = [a["text"] for a in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        collection.add(
            ids=[a["pmid"] for a in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "pmid": a["pmid"],
                    "title": a["title"],
                    "authors": a["authors"],
                    "year": a["year"],
                    "journal": a["journal"],
                }
                for a in batch
            ],
        )

    total = collection.count()
    print(f"\nDone. Added {len(all_articles)} articles. Total in DB: {total}")
    return len(all_articles)


def _chunk_text(text: str, chunk_size: int = 800) -> list[str]:
    """Split text into chunks of ~chunk_size chars, preferring paragraph breaks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                # paragraph too long — split naively at sentence boundaries
                sentences = para.replace(". ", ".\n").split("\n")
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= chunk_size:
                        current = (current + " " + sent).strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def _oir_csv_to_text(path: Path) -> str:
    """Convert OIR.csv rows to embeddable natural-language sentences."""
    import csv as _csv

    lines: list[str] = []
    with open(path, newline="", encoding="utf-8-sig", errors="ignore") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            name = row.get("Names", "").strip()
            condition = row.get("NOX_vs_OIR", "").strip()
            age_sac = row.get("Age_at_Sac", "").strip()
            tx_age = row.get("Treatment age", "").strip()
            treatment = row.get("Treatment", "").strip()
            dose = row.get("Dose", "").strip()
            if not name:
                continue
            parts = [f"Retinal flatmount sample: {name}."]
            if condition:
                parts.append(f"Condition: {condition}.")
            if age_sac:
                parts.append(f"Sacrificed at postnatal day P{age_sac}.")
            if treatment:
                tx_str = treatment
                if dose:
                    tx_str += f" (dose: {dose})"
                if tx_age:
                    tx_str += f" administered at P{tx_age}"
                parts.append(f"Treatment: {tx_str}.")
            lines.append(" ".join(parts))
    return "\n\n".join(lines)


def ingest_local(
    paths: list[Path | str],
    chunk_size: int = 800,
    model: SentenceTransformer | None = None,
) -> int:
    """Chunk and embed local .txt / .pdf files into ChromaDB.

    IDs are content-addressed so re-ingesting the same file is safe (no duplicates).
    Returns the number of newly added chunks.
    """
    if model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    existing_ids = set(collection.get(include=[])["ids"])

    all_chunks: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            try:
                import fitz  # pymupdf

                doc = fitz.open(str(path))
                text = "\n\n".join(page.get_text() for page in doc)
            except ImportError:
                print(
                    f"  Skipping {path.name}: install pymupdf for PDF support "
                    "(`pip install pymupdf`)"
                )
                continue
        elif suffix == ".csv":
            text = _oir_csv_to_text(path)
        else:
            print(f"  Skipping {path.name}: unsupported format (use .txt, .pdf, or .csv)")
            continue

        chunks = _chunk_text(text, chunk_size)
        print(f"  {path.name}: {len(chunks)} chunk(s)")

        for ci, chunk in enumerate(chunks):
            chunk_id = (
                "local_" + hashlib.md5((path.name + str(ci) + chunk).encode()).hexdigest()[:16]
            )
            if chunk_id in existing_ids:
                continue
            all_chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "pmid": "",
                        "title": path.stem.replace("_", " ").replace("-", " ").title(),
                        "authors": "",
                        "year": "",
                        "journal": "Local",
                        "source": "local",
                        "filename": path.name,
                    },
                }
            )

    if not all_chunks:
        print("Nothing new to add.")
        return 0

    print(f"\nEmbedding {len(all_chunks)} new chunk(s)…")
    batch_size = 64
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
        batch = all_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c["metadata"] for c in batch],
        )

    total = collection.count()
    print(f"Done. Added {len(all_chunks)} chunk(s). Total in DB: {total}")
    return len(all_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch PubMed abstracts and embed into local ChromaDB."
    )
    parser.add_argument(
        "queries",
        nargs="*",
        default=None,
        help="Search queries (default: retina/ROP focused set from config.py)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=MAX_RESULTS_PER_QUERY,
        help=f"Max results per query (default: {MAX_RESULTS_PER_QUERY})",
    )
    parser.add_argument(
        "--local",
        nargs="+",
        metavar="PATH",
        help="Local .txt / .pdf files (or folders) to ingest instead of querying PubMed",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Characters per chunk for local files (default: 800)",
    )
    args = parser.parse_args()

    if args.local:
        paths: list[Path] = []
        for p in args.local:
            path = Path(p)
            if path.is_dir():
                paths.extend(
                    f for f in path.iterdir() if f.suffix.lower() in (".txt", ".pdf", ".csv")
                )
            else:
                paths.append(path)
        ingest_local(paths, chunk_size=args.chunk_size)
    else:
        queries = args.queries if args.queries else DEFAULT_QUERIES
        ingest(queries, max_per_query=args.max)


if __name__ == "__main__":
    main()
