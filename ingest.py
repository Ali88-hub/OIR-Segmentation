"""Fetch PubMed abstracts (and PMC full text) and embed them into a local ChromaDB collection.

Usage:
    python ingest.py                              # default retina/ROP queries
    python ingest.py "vaso-obliteration" "VEGF"   # custom queries
    python ingest.py --max 500                    # fetch more per query
    python ingest.py --no-fulltext                # skip PMC full-text, abstracts only
"""

from __future__ import annotations

import argparse
import hashlib
import time
import xml.etree.ElementTree as ET
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
    FULLTEXT_CHUNK_SIZE,
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


def fetch_pmcids(pmids: list[str]) -> dict[str, str]:
    """Map PubMed IDs to PubMed Central IDs via Entrez elink.

    Returns a dict of {pmid: pmcid} for articles available in PMC.
    """
    if not pmids:
        return {}

    mapping: dict[str, str] = {}
    batch_size = 100
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        try:
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pmc",
                id=batch,
                linkname="pubmed_pmc",
            )
            records = Entrez.read(handle)
            handle.close()
        except Exception as exc:
            print(f"  Warning: elink failed for batch — {exc}")
            continue

        for record in records:
            pm_id = str(record["IdList"][0]) if record["IdList"] else ""
            if not pm_id:
                continue
            link_sets = record.get("LinkSetDb", [])
            for ls in link_sets:
                if ls.get("LinkName") == "pubmed_pmc":
                    for link in ls.get("Link", []):
                        mapping[pm_id] = str(link["Id"])
                        break  # take first PMC link

        time.sleep(0.35 if not NCBI_API_KEY else 0.11)

    return mapping


def _extract_text(elem: ET.Element) -> str:
    """Recursively extract all text content from an XML element, including tails."""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_extract_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _parse_pmc_xml(xml_bytes: bytes, pmid_to_pmcid: dict[str, str]) -> list[dict]:
    """Parse PMC JATS XML and extract structured sections.

    Returns list of article dicts with 'sections' field.
    """
    # Reverse mapping: pmcid -> pmid
    pmcid_to_pmid = {v: k for k, v in pmid_to_pmcid.items()}

    articles = []
    # PMC efetch returns <pmc-articleset> with multiple <article> elements
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        print("  Warning: failed to parse PMC XML")
        return []

    # Handle both single article and article-set
    if root.tag == "article":
        article_elems = [root]
    else:
        article_elems = root.findall(".//article")

    for art_elem in article_elems:
        try:
            # Extract PMC ID
            front = art_elem.find(".//front")
            if front is None:
                continue

            article_meta = front.find(".//article-meta")
            if article_meta is None:
                continue

            pmcid = ""
            pmid = ""
            for aid in article_meta.findall(".//article-id"):
                if aid.get("pub-id-type") == "pmc":
                    pmcid = (aid.text or "").strip()
                elif aid.get("pub-id-type") == "pmid":
                    pmid = (aid.text or "").strip()

            if not pmcid and not pmid:
                continue

            # Resolve PMID<->PMCID
            if not pmid:
                pmid = pmcid_to_pmid.get(pmcid, "")
            if not pmcid:
                pmcid = pmid_to_pmcid.get(pmid, "")

            # Title
            title_elem = article_meta.find(".//article-title")
            title = _extract_text(title_elem).strip() if title_elem is not None else ""

            # Authors
            authors_parsed = []
            for contrib in article_meta.findall(".//contrib[@contrib-type='author']"):
                surname = contrib.findtext(".//surname", "")
                given = contrib.findtext(".//given-names", "")
                if surname:
                    authors_parsed.append(f"{surname} {given}".strip())
            authors = ", ".join(authors_parsed[:3])
            if len(authors_parsed) > 3:
                authors += " et al."

            # Year
            pub_date = article_meta.find(".//pub-date")
            year = ""
            if pub_date is not None:
                year = pub_date.findtext("year", "")

            # Journal
            journal_meta = front.find(".//journal-meta")
            journal = ""
            if journal_meta is not None:
                journal = journal_meta.findtext(".//journal-title", "")

            # Abstract
            sections: list[dict[str, str]] = []
            abstract_elem = article_meta.find(".//abstract")
            if abstract_elem is not None:
                abstract_text = _extract_text(abstract_elem).strip()
                if abstract_text:
                    sections.append({"name": "Abstract", "text": abstract_text})

            # Body sections
            body = art_elem.find(".//body")
            if body is not None:
                for sec in body.findall("sec"):
                    sec_title_elem = sec.find("title")
                    sec_name = (
                        _extract_text(sec_title_elem).strip()
                        if sec_title_elem is not None
                        else "Body"
                    )
                    sec_text = _extract_text(sec).strip()
                    # Remove the section title from the beginning of the text
                    if sec_name and sec_text.startswith(sec_name):
                        sec_text = sec_text[len(sec_name) :].strip()
                    if sec_text:
                        sections.append({"name": sec_name, "text": sec_text})

            if not sections:
                continue

            articles.append(
                {
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "journal": journal,
                    "sections": sections,
                }
            )
        except (KeyError, IndexError, AttributeError):
            continue

    return articles


def fetch_fulltext(pmcids: list[str], pmid_to_pmcid: dict[str, str]) -> list[dict]:
    """Fetch full-text JATS XML from PMC for a list of PMC IDs.

    Returns list of article dicts with structured 'sections' field.
    """
    if not pmcids:
        return []

    all_articles: list[dict] = []
    batch_size = 20  # PMC returns large XML — smaller batches
    for i in range(0, len(pmcids), batch_size):
        batch = pmcids[i : i + batch_size]
        try:
            handle = Entrez.efetch(
                db="pmc",
                id=",".join(batch),
                rettype="full",
                retmode="xml",
            )
            xml_bytes = handle.read()
            handle.close()
        except Exception as exc:
            print(f"  Warning: PMC efetch failed for batch — {exc}")
            continue

        parsed = _parse_pmc_xml(xml_bytes, pmid_to_pmcid)
        all_articles.extend(parsed)
        time.sleep(0.35 if not NCBI_API_KEY else 0.11)

    return all_articles


def _normalize_section_name(name: str) -> str:
    """Normalize section titles to canonical names for metadata."""
    lower = name.lower().strip()
    if "abstract" in lower:
        return "abstract"
    if "intro" in lower:
        return "introduction"
    if "method" in lower or "material" in lower:
        return "methods"
    if "result" in lower:
        return "results"
    if "discuss" in lower:
        return "discussion"
    if "conclu" in lower:
        return "conclusion"
    return lower


def _chunk_sections(article: dict, chunk_size: int = FULLTEXT_CHUNK_SIZE) -> list[dict]:
    """Chunk a full-text article by section, returning embeddable chunks with metadata."""
    chunks: list[dict] = []
    pmcid = article["pmcid"]
    pmid = article["pmid"]

    for si, section in enumerate(article["sections"]):
        sec_name = section["name"]
        sec_normalized = _normalize_section_name(sec_name)
        text_chunks = _chunk_text(section["text"], chunk_size)

        for ci, chunk_text in enumerate(text_chunks):
            # Content-addressed ID (same pattern as ingest_local)
            chunk_id = (
                "pmc_"
                + hashlib.md5((pmcid + str(si) + str(ci) + chunk_text).encode()).hexdigest()[:16]
            )
            # Prefix with section name for embedding context
            embedded_text = f"[{sec_name}] {chunk_text}"

            chunks.append(
                {
                    "id": chunk_id,
                    "text": embedded_text,
                    "metadata": {
                        "pmid": pmid,
                        "pmcid": pmcid,
                        "title": article["title"],
                        "authors": article["authors"],
                        "year": article["year"],
                        "journal": article["journal"],
                        "section": sec_normalized,
                        "source": "pmc",
                    },
                }
            )

    return chunks


def ingest(
    queries: list[str],
    max_per_query: int = MAX_RESULTS_PER_QUERY,
    model: SentenceTransformer | None = None,
    fulltext: bool = True,
) -> int:
    """Fetch papers for all queries, embed, and store in ChromaDB.

    When *fulltext* is True (default), articles available in PubMed Central
    are fetched as full-text and chunked by section.  Articles without PMC
    availability fall back to abstract-only ingestion.

    Returns the number of newly added documents/chunks.
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

    # Items to embed: each has 'id', 'text', 'metadata'
    embed_items: list[dict] = []
    seen_pmids: set[str] = set()
    # Also track existing PMIDs (IDs that are raw PMIDs, not chunk IDs)
    for eid in existing_ids:
        if eid.isdigit():
            seen_pmids.add(eid)

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

        if not new_pmids:
            continue

        # --- Full-text path: resolve PMC IDs and fetch where available ---
        pmids_for_abstract = list(new_pmids)  # default: all go to abstract path

        if fulltext:
            print("  Resolving PMC availability…")
            pmid_to_pmcid = fetch_pmcids(new_pmids)
            pmc_pmids = [p for p in new_pmids if p in pmid_to_pmcid]
            pmids_for_abstract = [p for p in new_pmids if p not in pmid_to_pmcid]
            print(f"  {len(pmc_pmids)} available in PMC, {len(pmids_for_abstract)} abstract-only")

            if pmc_pmids:
                pmcids = [pmid_to_pmcid[p] for p in pmc_pmids]
                ft_articles = fetch_fulltext(pmcids, pmid_to_pmcid)
                print(f"  Parsed {len(ft_articles)} full-text article(s)")
                for art in ft_articles:
                    chunks = _chunk_sections(art)
                    for chunk in chunks:
                        if chunk["id"] not in existing_ids:
                            embed_items.append(chunk)

        # --- Abstract-only path (unchanged logic) ---
        for i in range(0, len(pmids_for_abstract), 100):
            batch_pmids = pmids_for_abstract[i : i + 100]
            try:
                articles = fetch_abstracts(batch_pmids)
            except Exception as exc:
                print(f"  Warning: failed to fetch abstracts for batch — {exc}")
                continue
            for a in articles:
                if a["pmid"] not in existing_ids:
                    embed_items.append(
                        {
                            "id": a["pmid"],
                            "text": a["text"],
                            "metadata": {
                                "pmid": a["pmid"],
                                "pmcid": "",
                                "title": a["title"],
                                "authors": a["authors"],
                                "year": a["year"],
                                "journal": a["journal"],
                                "section": "abstract",
                                "source": "pubmed",
                            },
                        }
                    )
            time.sleep(0.35 if not NCBI_API_KEY else 0.11)

    if not embed_items:
        print("\nNothing new to add.")
        return 0

    print(f"\nEmbedding {len(embed_items)} new item(s) …")

    batch_size = 64
    for i in tqdm(range(0, len(embed_items), batch_size), desc="Embedding"):
        batch = embed_items[i : i + batch_size]
        texts = [item["text"] for item in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        collection.add(
            ids=[item["id"] for item in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[item["metadata"] for item in batch],
        )

    total = collection.count()
    print(f"\nDone. Added {len(embed_items)} item(s). Total in DB: {total}")
    return len(embed_items)


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
        description="Fetch PubMed abstracts (and PMC full text) and embed into local ChromaDB."
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
    fulltext_group = parser.add_mutually_exclusive_group()
    fulltext_group.add_argument(
        "--fulltext",
        action="store_true",
        default=True,
        help="Fetch full text from PMC when available (default)",
    )
    fulltext_group.add_argument(
        "--no-fulltext",
        dest="fulltext",
        action="store_false",
        help="Skip PMC full text — fetch abstracts only",
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
        ingest(queries, max_per_query=args.max, fulltext=args.fulltext)


if __name__ == "__main__":
    main()
