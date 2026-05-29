from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any

import numpy as np
from neo4j import Driver, GraphDatabase
from sentence_transformers import SentenceTransformer

from app.rag_service import RetrievedChunk


@dataclass(frozen=True)
class GraphEntityEvidence:
    company: str
    entity_name: str
    entity_type: str
    entity_description: str
    degree: int
    frequency: int
    source_refs: list[str]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphRelationshipEvidence:
    company: str
    source_entity: str
    target_entity: str
    relationship_description: str
    relationship_weight: float
    relationship_strength_max: int
    occurrence_count: int
    combined_degree: int
    source_refs: list[str]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompanyGraphEvidence:
    entities: list[GraphEntityEvidence]
    relationships: list[GraphRelationshipEvidence]
    chunks: list[RetrievedChunk]


@dataclass
class _CompanyGraphIndex:
    entities: list[dict[str, Any]]
    entity_embeddings: np.ndarray
    relationships: list[dict[str, Any]]


def _dot(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _normalize_companies(companies: list[str]) -> list[str]:
    normalized = [company.strip().lower() for company in companies if company.strip()]
    return list(dict.fromkeys(normalized))


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            output.append(text)
    return output


def _parse_source_ref(source_ref: str) -> tuple[str, int] | None:
    parts = str(source_ref).rsplit(":", 2)
    if len(parts) < 2:
        return None

    source_file = parts[0].strip()
    if not source_file:
        return None

    try:
        page_number = int(parts[-2])
    except ValueError:
        return None

    return source_file, page_number


def _entity_embedding_text(row: dict[str, Any]) -> str:
    name = str(row.get("entity_name", "")).strip()
    entity_type = str(row.get("entity_type", "")).strip()
    description = str(row.get("entity_description", "")).strip()
    degree = _to_int(row.get("degree", 0), 0)

    return (
        f"Entity: {name}. Type: {entity_type}. "
        f"Description: {description}. Degree: {degree}."
    )


class GraphRagService:
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

        self._graph_cache: dict[str, _CompanyGraphIndex] = {}
        self._graph_cache_lock = Lock()

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

    def _embed_text(self, text: str) -> list[float]:
        model = self._get_embedding_model()
        vector = model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()

    def list_companies(self) -> list[str]:
        query = """
        MATCH (company:Company)
        RETURN company.name AS name
        ORDER BY name
        """

        with self._driver.session(database=self._database) as session:
            rows = session.run(query).data()

        names = [str(row.get("name", "")).strip() for row in rows if row.get("name")]
        return _normalize_companies(names)

    def _fetch_company_entities(self, company: str) -> list[dict[str, Any]]:
        query = """
        MATCH (entity:Entity {company: $company})
        RETURN
            entity.company AS company,
            entity.name AS entity_name,
            entity.type AS entity_type,
            entity.description AS entity_description,
            entity.degree AS degree,
            entity.frequency AS frequency,
            entity.source_refs AS source_refs
        """

        with self._driver.session(database=self._database) as session:
            return session.run(query, company=company).data()

    def _fetch_company_relationships(self, company: str) -> list[dict[str, Any]]:
        query = """
        MATCH (source:Entity {company: $company})-[rel:RELATED_TO {company: $company}]->(target:Entity {company: $company})
        RETURN
            rel.company AS company,
            source.name AS source_entity,
            target.name AS target_entity,
            rel.description AS relationship_description,
            rel.weight AS relationship_weight,
            rel.strength_max AS relationship_strength_max,
            rel.occurrence_count AS occurrence_count,
            rel.combined_degree AS combined_degree,
            rel.source_refs AS source_refs
        """

        with self._driver.session(database=self._database) as session:
            return session.run(query, company=company).data()

    def _load_company_graph_index(self, company: str) -> _CompanyGraphIndex:
        with self._graph_cache_lock:
            cached = self._graph_cache.get(company)
            if cached is not None:
                return cached

        entities = self._fetch_company_entities(company)
        relationships = self._fetch_company_relationships(company)

        if entities:
            model = self._get_embedding_model()
            texts = [_entity_embedding_text(row) for row in entities]
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        else:
            embeddings = np.empty((0, 0), dtype=float)

        index = _CompanyGraphIndex(
            entities=entities,
            entity_embeddings=embeddings,
            relationships=relationships,
        )

        with self._graph_cache_lock:
            self._graph_cache[company] = index

        return index

    def _fetch_chunks_by_refs(
        self,
        company: str,
        refs: list[dict[str, int | str]],
        page_importance: dict[tuple[str, int], float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not refs or top_k < 1:
            return []

        query = """
        UNWIND $refs AS ref
        MATCH (chunk:Chunk {
            company: $company,
            source_file: ref.source_file,
            page_number: ref.page_number
        })
        RETURN DISTINCT
            chunk.id AS id,
            chunk.company AS company,
            chunk.source_file AS source_file,
            chunk.page_number AS page_number,
            chunk.page_number_raw AS page_number_raw,
            chunk.page_ordinal AS page_ordinal,
            chunk.text AS text
        """

        with self._driver.session(database=self._database) as session:
            rows = session.run(query, company=company, refs=refs).data()

        scored_rows: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            source_file = str(row.get("source_file", "")).strip()
            page_number = _to_int(row.get("page_number"), -1)
            importance = page_importance.get((source_file, page_number), 0.0)
            scored_rows.append((importance, row))

        scored_rows.sort(
            key=lambda item: (
                -item[0],
                str(item[1].get("source_file", "")),
                _to_int(item[1].get("page_number"), -1),
            )
        )

        chunks: list[RetrievedChunk] = []
        for importance, row in scored_rows[:top_k]:
            chunks.append(
                RetrievedChunk(
                    id=str(row.get("id", "")),
                    company=str(row.get("company", company)),
                    source_file=str(row.get("source_file", "")),
                    page_number=_to_int(row.get("page_number"), -1),
                    page_number_raw=str(row.get("page_number_raw", "")),
                    page_ordinal=_to_int(row.get("page_ordinal"), -1),
                    score=float(importance),
                    text=str(row.get("text", "")),
                )
            )
        return chunks

    def retrieve_by_company(
        self,
        question: str,
        companies: list[str],
        top_entities_per_company: int,
        top_relationships_per_company: int,
        top_chunks_per_company: int,
    ) -> dict[str, CompanyGraphEvidence]:
        if top_entities_per_company < 1:
            raise ValueError("top_entities_per_company must be >= 1")
        if top_relationships_per_company < 1:
            raise ValueError("top_relationships_per_company must be >= 1")
        if top_chunks_per_company < 1:
            raise ValueError("top_chunks_per_company must be >= 1")

        normalized_companies = _normalize_companies(companies)
        if not normalized_companies:
            normalized_companies = self.list_companies()

        if not normalized_companies:
            return {}

        question_embedding = self._embed_text(question)
        results: dict[str, CompanyGraphEvidence] = {}

        for company in normalized_companies:
            index = self._load_company_graph_index(company)

            if not index.entities:
                results[company] = CompanyGraphEvidence(
                    entities=[], relationships=[], chunks=[]
                )
                continue

            entity_scores: list[tuple[float, dict[str, Any]]] = []

            if index.entity_embeddings.size > 0:
                question_vec = np.array(question_embedding, dtype=float)
                scores = np.dot(index.entity_embeddings, question_vec)
                for entity_row, score in zip(index.entities, scores):
                    entity_scores.append((float(score), entity_row))
            else:
                for entity_row in index.entities:
                    embedding_text = _entity_embedding_text(entity_row)
                    score = _dot(question_embedding, self._embed_text(embedding_text))
                    entity_scores.append((score, entity_row))

            entity_scores.sort(key=lambda item: item[0], reverse=True)
            selected_entity_scores = entity_scores[:top_entities_per_company]

            entity_evidence: list[GraphEntityEvidence] = []
            seed_scores: dict[str, float] = {}

            for score, row in selected_entity_scores:
                entity_name = str(row.get("entity_name", "")).strip()
                if not entity_name:
                    continue

                normalized_score = float(max(0.0, score))
                seed_scores[entity_name] = normalized_score
                entity_evidence.append(
                    GraphEntityEvidence(
                        company=str(row.get("company", company)),
                        entity_name=entity_name,
                        entity_type=str(row.get("entity_type", "")),
                        entity_description=str(row.get("entity_description", "")),
                        degree=_to_int(row.get("degree"), 0),
                        frequency=_to_int(row.get("frequency"), 0),
                        source_refs=_ensure_string_list(row.get("source_refs")),
                        score=round(normalized_score, 4),
                    )
                )

            seed_entities = set(seed_scores.keys())
            selected_relationships: list[GraphRelationshipEvidence] = []
            relationship_candidates: list[tuple[int, float, dict[str, Any]]] = []

            for relationship_row in index.relationships:
                source_entity = str(relationship_row.get("source_entity", "")).strip()
                target_entity = str(relationship_row.get("target_entity", "")).strip()
                if not source_entity or not target_entity:
                    continue

                seed_hits = int(source_entity in seed_entities) + int(
                    target_entity in seed_entities
                )
                weight = _to_float(relationship_row.get("relationship_weight"), 0.0)
                combined_degree = _to_int(relationship_row.get("combined_degree"), 0)
                seed_boost = seed_scores.get(source_entity, 0.0) + seed_scores.get(
                    target_entity, 0.0
                )

                score = (
                    weight
                    + (0.03 * float(combined_degree))
                    + seed_boost
                    + (seed_hits * 2.0)
                )
                relationship_candidates.append((seed_hits, score, relationship_row))

            relationship_candidates.sort(
                key=lambda item: (item[0], item[1]), reverse=True
            )

            for seed_hits, score, row in relationship_candidates[
                :top_relationships_per_company
            ]:
                selected_relationships.append(
                    GraphRelationshipEvidence(
                        company=str(row.get("company", company)),
                        source_entity=str(row.get("source_entity", "")),
                        target_entity=str(row.get("target_entity", "")),
                        relationship_description=str(
                            row.get("relationship_description", "")
                        ),
                        relationship_weight=_to_float(
                            row.get("relationship_weight"), 0.0
                        ),
                        relationship_strength_max=_to_int(
                            row.get("relationship_strength_max"), 0
                        ),
                        occurrence_count=_to_int(row.get("occurrence_count"), 0),
                        combined_degree=_to_int(row.get("combined_degree"), 0),
                        source_refs=_ensure_string_list(row.get("source_refs")),
                        score=round(max(0.0, score), 4),
                    )
                )

            page_importance: dict[tuple[str, int], float] = {}

            for entity in entity_evidence:
                for source_ref in entity.source_refs:
                    parsed = _parse_source_ref(source_ref)
                    if parsed is None:
                        continue
                    source_file, page_number = parsed
                    key = (source_file, page_number)
                    page_importance[key] = (
                        page_importance.get(key, 0.0) + 1.0 + entity.score
                    )

            for relationship in selected_relationships:
                for source_ref in relationship.source_refs:
                    parsed = _parse_source_ref(source_ref)
                    if parsed is None:
                        continue
                    source_file, page_number = parsed
                    key = (source_file, page_number)
                    page_importance[key] = (
                        page_importance.get(key, 0.0)
                        + 2.0
                        + min(relationship.score, 10.0)
                    )

            refs = [
                {"source_file": source_file, "page_number": page_number}
                for source_file, page_number in sorted(page_importance.keys())
            ]
            chunks = self._fetch_chunks_by_refs(
                company=company,
                refs=refs,
                page_importance=page_importance,
                top_k=top_chunks_per_company,
            )

            results[company] = CompanyGraphEvidence(
                entities=entity_evidence,
                relationships=selected_relationships,
                chunks=chunks,
            )

        return results
