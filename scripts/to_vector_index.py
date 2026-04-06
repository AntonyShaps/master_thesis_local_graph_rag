from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from neo4j import Driver, GraphDatabase
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import DATA_MARKDOWN_DIR

PAGE_TAG_RE = re.compile(r"<page_number>\s*([0-9]+)\s*</page_number>", re.IGNORECASE)
DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_VECTOR_INDEX_NAME = "chunk_embedding_index"
EMBEDDING_DIMENSIONS = 768
OVERLAP_RATIO = 0.5


@dataclass(frozen=True)
class PageSection:
    company: str
    source_file: str
    page_number_raw: str
    page_number: int
    page_ordinal: int
    text: str


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    company: str
    source_file: str
    page_number_raw: str
    page_number: int
    page_ordinal: int
    text: str


def _resolve_neo4j_auth() -> tuple[str, str]:
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not user or not password:
        auth_pair = os.getenv("NEO4J_AUTH")
        if auth_pair and "/" in auth_pair:
            split_user, split_password = auth_pair.split("/", 1)
            user = user or split_user
            password = password or split_password

    return user or "neo4j", password or "testpassword"


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _first_fraction_by_words(text: str, fraction: float) -> str:
    words = text.split()
    if not words:
        return ""
    count = max(1, int(len(words) * fraction))
    return " ".join(words[:count]).strip()


def _last_fraction_by_words(text: str, fraction: float) -> str:
    words = text.split()
    if not words:
        return ""
    count = max(1, int(len(words) * fraction))
    return " ".join(words[-count:]).strip()


def parse_markdown_pages(company: str, md_file: Path) -> list[PageSection]:
    content = md_file.read_text(encoding="utf-8")
    matches = list(PAGE_TAG_RE.finditer(content))

    pages: list[PageSection] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        page_text = _normalize_text(content[start:end])
        page_number_raw = match.group(1)
        page_number = int(page_number_raw)

        pages.append(
            PageSection(
                company=company,
                source_file=md_file.name,
                page_number_raw=page_number_raw,
                page_number=page_number,
                page_ordinal=i + 1,
                text=page_text,
            )
        )

    return pages


def build_chunks_from_pages(pages: list[PageSection]) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []

    for i, page in enumerate(pages):
        prev_tail = (
            _last_fraction_by_words(pages[i - 1].text, OVERLAP_RATIO) if i > 0 else ""
        )
        next_head = (
            _first_fraction_by_words(pages[i + 1].text, OVERLAP_RATIO)
            if i + 1 < len(pages)
            else ""
        )

        merged_text = "\n\n".join(
            [part for part in [prev_tail, page.text, next_head] if part]
        ).strip()
        if not merged_text:
            continue

        source_stem = Path(page.source_file).stem
        chunk_id = f"{page.company}:{source_stem}:{page.page_ordinal:04d}:{page.page_number_raw}"

        chunks.append(
            ChunkRecord(
                id=chunk_id,
                company=page.company,
                source_file=page.source_file,
                page_number_raw=page.page_number_raw,
                page_number=page.page_number,
                page_ordinal=page.page_ordinal,
                text=merged_text,
            )
        )

    return chunks


def collect_chunks(markdown_dir: Path) -> list[ChunkRecord]:
    all_chunks: list[ChunkRecord] = []
    for md_file in sorted(markdown_dir.glob("*.md")):
        company = md_file.stem.lower().strip()
        pages = parse_markdown_pages(company=company, md_file=md_file)
        chunks = build_chunks_from_pages(pages)
        all_chunks.extend(chunks)
        print(
            f"Parsed {md_file.name}: {len(pages)} tagged pages -> {len(chunks)} chunks"
        )

    return all_chunks


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _validate_identifier(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(
            f"Invalid identifier: {name}. Use letters, numbers, and underscore only."
        )


def create_schema(driver: Driver, database: str, vector_index_name: str) -> None:
    _validate_identifier(vector_index_name)

    create_company_constraint = (
        "CREATE CONSTRAINT company_name_unique IF NOT EXISTS "
        "FOR (company:Company) REQUIRE company.name IS UNIQUE"
    )
    create_chunk_constraint = (
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
        "FOR (chunk:Chunk) REQUIRE chunk.id IS UNIQUE"
    )
    create_vector_index = (
        f"CREATE VECTOR INDEX {vector_index_name} IF NOT EXISTS "
        "FOR (chunk:Chunk) ON (chunk.embedding) "
        "OPTIONS {indexConfig: {"
        f"`vector.dimensions`: {EMBEDDING_DIMENSIONS}, "
        "`vector.similarity_function`: 'cosine'"
        "}}"
    )

    with driver.session(database=database) as session:
        session.run(create_company_constraint).consume()
        session.run(create_chunk_constraint).consume()
        session.run(create_vector_index).consume()


def delete_company_chunks(driver: Driver, database: str, companies: list[str]) -> None:
    if not companies:
        return

    query = """
    UNWIND $companies AS company
    MATCH (chunk:Chunk {company: company})
    DETACH DELETE chunk
    """

    with driver.session(database=database) as session:
        session.run(query, companies=companies).consume()


def upsert_chunks(
    driver: Driver, database: str, rows: list[dict], batch_size: int
) -> None:
    query = """
    UNWIND $rows AS row
    MERGE (company:Company {name: row.company})
    MERGE (chunk:Chunk {id: row.id})
    SET chunk.company = row.company,
        chunk.source_file = row.source_file,
        chunk.page_number_raw = row.page_number_raw,
        chunk.page_number = row.page_number,
        chunk.page_ordinal = row.page_ordinal,
        chunk.text = row.text,
        chunk.embedding = row.embedding
    MERGE (company)-[:HAS_CHUNK]->(chunk)
    """

    with driver.session(database=database) as session:
        for i, batch in enumerate(batched(rows, batch_size), start=1):
            session.run(query, rows=batch).consume()
            print(f"Inserted batch {i} ({len(batch)} chunks)")


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    show_progress_bar: bool,
) -> list[list[float]]:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [vector.tolist() for vector in vectors]


def vector_dot(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def fetch_company_chunks(driver: Driver, database: str, company: str) -> list[dict]:
    query = """
    MATCH (chunk:Chunk {company: $company})
    RETURN
        chunk.id AS id,
        chunk.company AS company,
        chunk.source_file AS source_file,
        chunk.page_number AS page_number,
        chunk.page_number_raw AS page_number_raw,
        chunk.page_ordinal AS page_ordinal,
        chunk.text AS text,
        chunk.embedding AS embedding
    """

    with driver.session(database=database) as session:
        records = session.run(query, company=company).data()
    return records


def retrieve_similar_chunks_by_company(
    driver: Driver,
    database: str,
    model: SentenceTransformer,
    question: str,
    companies: list[str],
    top_k_per_company: int,
) -> dict[str, list[dict]]:
    normalized_companies = [company.lower().strip() for company in companies]
    unique_companies = list(
        dict.fromkeys(company for company in normalized_companies if company)
    )

    if not unique_companies:
        raise ValueError("At least one non-empty company name is required.")

    question_embedding = encode_texts(
        model=model,
        texts=[question],
        batch_size=1,
        show_progress_bar=False,
    )[0]

    company_results: dict[str, list[dict]] = {}
    for company in unique_companies:
        rows = fetch_company_chunks(driver=driver, database=database, company=company)
        scored_rows: list[dict] = []

        for row in rows:
            embedding = row.pop("embedding", None)
            if not embedding:
                continue

            score = vector_dot(question_embedding, embedding)
            row["score"] = score
            scored_rows.append(row)

        scored_rows.sort(key=lambda row: row["score"], reverse=True)
        company_results[company] = scored_rows[:top_k_per_company]

    return company_results


def build_index(args: argparse.Namespace) -> None:
    markdown_dir = Path(args.markdown_dir)
    if not markdown_dir.exists():
        raise FileNotFoundError(f"Markdown directory does not exist: {markdown_dir}")

    chunks = collect_chunks(markdown_dir)
    if not chunks:
        raise RuntimeError(f"No chunks generated from markdown files in {markdown_dir}")

    model = SentenceTransformer(args.model_name)
    embeddings = encode_texts(
        model=model,
        texts=[chunk.text for chunk in chunks],
        batch_size=args.embed_batch_size,
        show_progress_bar=True,
    )

    if not embeddings:
        raise RuntimeError("Embedding model returned no vectors.")

    if len(embeddings[0]) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Expected embedding dimension {EMBEDDING_DIMENSIONS}, got {len(embeddings[0])}."
        )

    rows: list[dict] = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append(
            {
                "id": chunk.id,
                "company": chunk.company,
                "source_file": chunk.source_file,
                "page_number_raw": chunk.page_number_raw,
                "page_number": chunk.page_number,
                "page_ordinal": chunk.page_ordinal,
                "text": chunk.text,
                "embedding": embedding,
            }
        )

    driver = GraphDatabase.driver(
        args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password)
    )
    try:
        driver.verify_connectivity()
        create_schema(driver, args.neo4j_database, args.vector_index_name)

        if args.replace_existing:
            companies = sorted({chunk.company for chunk in chunks})
            print(f"Replacing existing chunks for companies: {', '.join(companies)}")
            delete_company_chunks(driver, args.neo4j_database, companies)

        upsert_chunks(
            driver, args.neo4j_database, rows=rows, batch_size=args.db_batch_size
        )
    finally:
        driver.close()

    print(f"Done. Indexed {len(rows)} chunks from {markdown_dir}.")


def run_query(args: argparse.Namespace) -> None:
    model = SentenceTransformer(args.model_name)
    driver = GraphDatabase.driver(
        args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password)
    )

    try:
        driver.verify_connectivity()
        results = retrieve_similar_chunks_by_company(
            driver=driver,
            database=args.neo4j_database,
            model=model,
            question=args.question,
            companies=args.companies,
            top_k_per_company=args.top_k,
        )
    finally:
        driver.close()

    payload = {
        "question": args.question,
        "companies": [company.lower().strip() for company in args.companies],
        "top_k_per_company": args.top_k,
        "results": results,
    }
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build Neo4j vector index from markdown ESG pages and retrieve "
            "company-scoped similar chunks."
        )
    )

    default_user, default_password = _resolve_neo4j_auth()
    default_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    default_database = os.getenv("NEO4J_DATABASE", "neo4j")

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser(
        "build", help="Create/update vector index in Neo4j"
    )
    build_cmd.add_argument(
        "--markdown-dir",
        default=str(DATA_MARKDOWN_DIR),
        help="Directory with markdown ESG files.",
    )
    build_cmd.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name for embeddings.",
    )
    build_cmd.add_argument(
        "--vector-index-name",
        default=DEFAULT_VECTOR_INDEX_NAME,
        help="Neo4j vector index name for chunk embeddings.",
    )
    build_cmd.add_argument(
        "--embed-batch-size",
        type=int,
        default=16,
        help="Batch size for embedding generation.",
    )
    build_cmd.add_argument(
        "--db-batch-size",
        type=int,
        default=100,
        help="Batch size for writing chunk rows to Neo4j.",
    )
    build_cmd.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete existing chunks for companies being indexed before insert.",
    )
    build_cmd.add_argument("--neo4j-uri", default=default_uri, help="Neo4j Bolt URI.")
    build_cmd.add_argument("--neo4j-user", default=default_user, help="Neo4j username.")
    build_cmd.add_argument(
        "--neo4j-password", default=default_password, help="Neo4j password."
    )
    build_cmd.add_argument(
        "--neo4j-database", default=default_database, help="Neo4j database name."
    )
    build_cmd.set_defaults(handler=build_index)

    query_cmd = subparsers.add_parser(
        "query", help="Retrieve similar chunks scoped to requested companies"
    )
    query_cmd.add_argument("--question", required=True, help="User question text.")
    query_cmd.add_argument(
        "--companies",
        nargs="+",
        required=True,
        help="Company names (same as markdown file stems, e.g. google nvidia).",
    )
    query_cmd.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top results returned per company.",
    )
    query_cmd.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name used for the question embedding.",
    )
    query_cmd.add_argument("--neo4j-uri", default=default_uri, help="Neo4j Bolt URI.")
    query_cmd.add_argument("--neo4j-user", default=default_user, help="Neo4j username.")
    query_cmd.add_argument(
        "--neo4j-password", default=default_password, help="Neo4j password."
    )
    query_cmd.add_argument(
        "--neo4j-database", default=default_database, help="Neo4j database name."
    )
    query_cmd.set_defaults(handler=run_query)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "embed_batch_size", 1) < 1:
        parser.error("--embed-batch-size must be >= 1")
    if getattr(args, "db_batch_size", 1) < 1:
        parser.error("--db-batch-size must be >= 1")
    if getattr(args, "top_k", 1) < 1:
        parser.error("--top-k must be >= 1")

    args.handler(args)


if __name__ == "__main__":
    main()
