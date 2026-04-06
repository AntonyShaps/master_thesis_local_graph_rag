from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock

from neo4j import Driver, GraphDatabase
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    company: str
    source_file: str
    page_number: int
    page_number_raw: str
    page_ordinal: int
    score: float
    text: str

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def _dot(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _normalize_companies(companies: list[str]) -> list[str]:
    normalized = [company.strip().lower() for company in companies if company.strip()]
    return list(dict.fromkeys(normalized))


class RagService:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str,
        embedding_model_name: str,
    ) -> None:
        self._driver: Driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )
        self._database = neo4j_database
        self._embedding_model_name = embedding_model_name
        self._embedding_model: SentenceTransformer | None = None
        self._embedding_lock = Lock()

    def close(self) -> None:
        self._driver.close()

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            with self._embedding_lock:
                if self._embedding_model is None:
                    self._embedding_model = SentenceTransformer(
                        self._embedding_model_name
                    )
        return self._embedding_model

    def _embed_question(self, question: str) -> list[float]:
        model = self._get_embedding_model()
        vector = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()

    def list_companies(self) -> list[str]:
        primary_query = """
        MATCH (company:Company)
        RETURN company.name AS name
        ORDER BY name
        """
        fallback_query = """
        MATCH (chunk:Chunk)
        WHERE chunk.company IS NOT NULL
        RETURN DISTINCT chunk.company AS name
        ORDER BY name
        """

        with self._driver.session(database=self._database) as session:
            records = session.run(primary_query).data()
            if not records:
                records = session.run(fallback_query).data()

        names = [
            str(record["name"]).strip() for record in records if record.get("name")
        ]
        return _normalize_companies(names)

    def _fetch_company_chunks(self, company: str) -> list[dict]:
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

        with self._driver.session(database=self._database) as session:
            return session.run(query, company=company).data()

    def retrieve_by_company(
        self,
        question: str,
        companies: list[str],
        top_k_per_company: int,
    ) -> dict[str, list[RetrievedChunk]]:
        if top_k_per_company < 1:
            raise ValueError("top_k_per_company must be >= 1")

        normalized_companies = _normalize_companies(companies)
        if not normalized_companies:
            normalized_companies = self.list_companies()

        if not normalized_companies:
            return {}

        question_embedding = self._embed_question(question)
        by_company: dict[str, list[RetrievedChunk]] = {}

        for company in normalized_companies:
            rows = self._fetch_company_chunks(company)
            scored: list[RetrievedChunk] = []

            for row in rows:
                embedding = row.get("embedding")
                if not isinstance(embedding, list) or not embedding:
                    continue

                score = _dot(question_embedding, [float(value) for value in embedding])
                scored.append(
                    RetrievedChunk(
                        id=str(row.get("id", "")),
                        company=str(row.get("company", company)),
                        source_file=str(row.get("source_file", "")),
                        page_number=(
                            int(row["page_number"])
                            if row.get("page_number") is not None
                            else -1
                        ),
                        page_number_raw=str(row.get("page_number_raw", "")),
                        page_ordinal=(
                            int(row["page_ordinal"])
                            if row.get("page_ordinal") is not None
                            else -1
                        ),
                        score=score,
                        text=str(row.get("text", "")),
                    )
                )

            scored.sort(key=lambda chunk: chunk.score, reverse=True)
            by_company[company] = scored[:top_k_per_company]

        return by_company
