from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from neo4j import Driver, GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENTITY_TYPES
from paths import DATA_MARKDOWN_DIR, MODELS_DIR

if TYPE_CHECKING:
    from app.mistral_service import MistralService


DEFAULT_GRAPH_EXTRACTION_DIR = DATA_MARKDOWN_DIR.parent / "graph_extraction"
DEFAULT_GRAPH_INDEX_DIR = DATA_MARKDOWN_DIR.parent / "graph_index"
DEFAULT_MODEL_DIR = Path(
    os.getenv("MISTRAL_MODEL_DIR", str(MODELS_DIR / "Mistral-7B-Instruct-v0.3"))
)
DEFAULT_SUMMARY_MAX_NEW_TOKENS = int(os.getenv("GRAPH_SUMMARY_MAX_NEW_TOKENS", "320"))
DEFAULT_SUMMARY_TEMPERATURE = float(os.getenv("GRAPH_SUMMARY_TEMPERATURE", "0.0"))
DEFAULT_SUMMARY_MAX_LENGTH_WORDS = int(
    os.getenv("GRAPH_SUMMARY_MAX_LENGTH_WORDS", "140")
)
DEFAULT_SUMMARY_CHECKPOINT_EVERY = 20

SUMMARY_PROMPT_TEMPLATE = """
Instructions for entity and relationship summarization
You are a helpful assistant responsible for generating a comprehensive summary of
the data provided below. Given one or two entities, and a list of descriptions, all
related to the same entity or group of entities. Please concatenate all of these into
a single, comprehensive description. Make sure to include information collected from
all the descriptions. If the provided descriptions are contradictory, please resolve the
contradictions and provide a single, coherent summary. Make sure it is written in
third person, and include the entity names so we have the full context.
Limit the final description length to {max_length} words.
#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
""".strip()

ENTITY_TYPE_PRIORITY = {
    entity_type: index for index, entity_type in enumerate(ENTITY_TYPES)
}


@dataclass
class CompanyArtifacts:
    company: str
    source_file: str
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    text_units: list[dict[str, Any]]
    summary: dict[str, Any]


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


def _create_mistral_service(
    model_dir: Path,
    default_max_new_tokens: int,
    default_temperature: float,
) -> "MistralService":
    from app.mistral_service import MistralService as RuntimeMistralService

    return RuntimeMistralService(
        model_dir=model_dir,
        default_max_new_tokens=default_max_new_tokens,
        default_temperature=default_temperature,
    )


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_entity_name(value: str) -> str:
    normalized = _normalize_whitespace(str(value))
    normalized = normalized.strip("`\"' ")
    normalized = normalized.strip(".,:;")
    return normalized.upper()


def _normalize_entity_type(value: str) -> str:
    normalized = _normalize_whitespace(str(value)).upper()
    if normalized in ENTITY_TYPE_PRIORITY:
        return normalized
    return normalized or "UNKNOWN"


def _normalize_description(value: str) -> str:
    normalized = _normalize_whitespace(str(value))
    return normalized.strip('"')


def _normalize_strength(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 1.0

    clipped = max(1.0, min(10.0, numeric))
    return int(round(clipped))


def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_text_unit_id(
    company: str,
    source_file: str,
    page_number: int,
    unit_index: int,
) -> str:
    return f"{company}:{source_file}:{page_number}:{unit_index}"


def _format_sources(
    source_keys: set[tuple[str, int, int]],
) -> tuple[list[dict[str, int | str]], list[str]]:
    ordered = sorted(source_keys, key=lambda item: (item[0], item[1], item[2]))
    sources = [
        {
            "source_file": source_file,
            "page_number": page_number,
            "unit_index": unit_index,
        }
        for source_file, page_number, unit_index in ordered
    ]
    source_refs = [
        f"{source_file}:{page_number}:{unit_index}"
        for source_file, page_number, unit_index in ordered
    ]
    return sources, source_refs


def _sort_descriptions(descriptions: set[str]) -> list[str]:
    return sorted(descriptions, key=lambda text: (-len(text), text))


def _pick_default_description(descriptions: list[str], fallback_subject: str) -> str:
    if descriptions:
        return descriptions[0]
    return f"{fallback_subject} appears in the source report text."


def _pick_entity_type(type_counts: Counter[str]) -> tuple[str, list[str]]:
    if not type_counts:
        return "UNKNOWN", []

    ranked = sorted(
        type_counts.items(),
        key=lambda item: (
            -item[1],
            ENTITY_TYPE_PRIORITY.get(item[0], len(ENTITY_TYPE_PRIORITY) + 1),
            item[0],
        ),
    )
    canonical_type = ranked[0][0]
    alternate_types = [entity_type for entity_type, _ in ranked[1:]]
    return canonical_type, alternate_types


def _list_companies(
    graph_extraction_dir: Path, requested: Sequence[str] | None
) -> list[str]:
    company_dirs = [
        item.name.lower()
        for item in graph_extraction_dir.iterdir()
        if item.is_dir() and (item / "records_raw.jsonl").exists()
    ]
    available = sorted(dict.fromkeys(company_dirs))

    if not requested:
        return available

    requested_set = {
        company.strip().lower() for company in requested if company.strip()
    }
    return [company for company in available if company in requested_set]


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _aggregate_company(
    company: str,
    rows: Sequence[dict[str, Any]],
) -> CompanyArtifacts:
    entity_store: dict[str, dict[str, Any]] = {}
    relationship_store: dict[tuple[str, str], dict[str, Any]] = {}
    text_units: dict[str, dict[str, Any]] = {}

    skipped_records = 0
    source_file = f"{company}.md"

    for row in rows:
        row_source_file = _normalize_whitespace(
            str(row.get("source_file", source_file))
        )
        if row_source_file:
            source_file = row_source_file

        page_number = _to_int(row.get("page_number"), -1)
        unit_index = _to_int(row.get("unit_index"), -1)
        text_unit_id = _build_text_unit_id(
            company, row_source_file, page_number, unit_index
        )

        if text_unit_id not in text_units:
            text_units[text_unit_id] = {
                "id": text_unit_id,
                "company": company,
                "source_file": row_source_file,
                "page_number": page_number,
                "unit_index": unit_index,
                "text": str(row.get("input_text", "")).strip(),
                "error": str(row.get("error", "")).strip(),
            }

        normalized_records = row.get("normalized_records")
        if not isinstance(normalized_records, list):
            continue

        for record in normalized_records:
            if not isinstance(record, dict):
                skipped_records += 1
                continue

            record_type = _normalize_whitespace(
                str(record.get("record_type", ""))
            ).lower()
            source_key = (row_source_file, page_number, unit_index)

            if record_type == "entity":
                entity_name = _normalize_entity_name(str(record.get("entity_name", "")))
                entity_type = _normalize_entity_type(str(record.get("entity_type", "")))
                entity_description = _normalize_description(
                    str(record.get("entity_description", ""))
                )

                if not entity_name:
                    skipped_records += 1
                    continue

                state = entity_store.setdefault(
                    entity_name,
                    {
                        "type_counts": Counter(),
                        "descriptions": set(),
                        "text_unit_ids": set(),
                        "sources": set(),
                    },
                )
                state["type_counts"][entity_type] += 1
                if entity_description:
                    state["descriptions"].add(entity_description)
                state["text_unit_ids"].add(text_unit_id)
                state["sources"].add(source_key)
                continue

            if record_type == "relationship":
                source_entity = _normalize_entity_name(
                    str(record.get("source_entity", ""))
                )
                target_entity = _normalize_entity_name(
                    str(record.get("target_entity", ""))
                )
                relationship_description = _normalize_description(
                    str(record.get("relationship_description", ""))
                )

                if not source_entity or not target_entity:
                    skipped_records += 1
                    continue

                strength = _normalize_strength(record.get("relationship_strength", 1))
                state = relationship_store.setdefault(
                    (source_entity, target_entity),
                    {
                        "descriptions": set(),
                        "strength_values": [],
                        "text_unit_ids": set(),
                        "sources": set(),
                    },
                )

                if relationship_description:
                    state["descriptions"].add(relationship_description)
                state["strength_values"].append(strength)
                state["text_unit_ids"].add(text_unit_id)
                state["sources"].add(source_key)
                continue

            skipped_records += 1

    entities: list[dict[str, Any]] = []
    entity_type_by_name: dict[str, str] = {}

    for entity_name, state in entity_store.items():
        canonical_type, alternate_types = _pick_entity_type(state["type_counts"])
        description_list = _sort_descriptions(state["descriptions"])
        sources, source_refs = _format_sources(state["sources"])

        entity_record = {
            "record_type": "entity",
            "company": company,
            "entity_name": entity_name,
            "entity_type": canonical_type,
            "alternate_types": alternate_types,
            "entity_description": _pick_default_description(
                description_list, entity_name
            ),
            "description_list": description_list,
            "frequency": len(state["text_unit_ids"]),
            "text_unit_ids": sorted(state["text_unit_ids"]),
            "sources": sources,
            "source_refs": source_refs,
            "source_count": len(source_refs),
        }
        entities.append(entity_record)
        entity_type_by_name[entity_name] = canonical_type

    orphan_relationships_filtered = 0
    relationships: list[dict[str, Any]] = []

    for (source_entity, target_entity), state in relationship_store.items():
        if (
            source_entity not in entity_type_by_name
            or target_entity not in entity_type_by_name
        ):
            orphan_relationships_filtered += 1
            continue

        descriptions = _sort_descriptions(state["descriptions"])
        strengths = [int(value) for value in state["strength_values"]] or [1]
        sources, source_refs = _format_sources(state["sources"])

        relationship_record = {
            "record_type": "relationship",
            "company": company,
            "source_entity": source_entity,
            "source_entity_type": entity_type_by_name[source_entity],
            "target_entity": target_entity,
            "target_entity_type": entity_type_by_name[target_entity],
            "relationship_description": _pick_default_description(
                descriptions, f"{source_entity} and {target_entity}"
            ),
            "description_list": descriptions,
            "relationship_strength_max": max(strengths),
            "relationship_weight": round(float(sum(strengths)), 2),
            "occurrence_count": len(strengths),
            "text_unit_ids": sorted(state["text_unit_ids"]),
            "sources": sources,
            "source_refs": source_refs,
            "source_count": len(source_refs),
        }
        relationships.append(relationship_record)

    degree_counter: Counter[str] = Counter()
    for relationship in relationships:
        degree_counter[relationship["source_entity"]] += 1
        degree_counter[relationship["target_entity"]] += 1

    for entity in entities:
        entity["degree"] = int(degree_counter.get(entity["entity_name"], 0))

    for relationship in relationships:
        relationship["combined_degree"] = int(
            degree_counter.get(relationship["source_entity"], 0)
            + degree_counter.get(relationship["target_entity"], 0)
        )

    entities = sorted(
        entities,
        key=lambda item: (
            str(item.get("entity_name", "")),
            str(item.get("entity_type", "")),
        ),
    )
    relationships = sorted(
        relationships,
        key=lambda item: (
            str(item.get("source_entity", "")),
            str(item.get("target_entity", "")),
            -float(item.get("relationship_weight", 0.0)),
        ),
    )
    text_unit_rows = sorted(
        text_units.values(),
        key=lambda item: (
            str(item.get("source_file", "")),
            int(item.get("page_number", -1)),
            int(item.get("unit_index", -1)),
        ),
    )

    summary = {
        "company": company,
        "source_file": source_file,
        "text_units_count": len(text_unit_rows),
        "entities_count": len(entities),
        "relationships_count": len(relationships),
        "orphan_relationships_filtered": orphan_relationships_filtered,
        "skipped_records": skipped_records,
    }

    return CompanyArtifacts(
        company=company,
        source_file=source_file,
        entities=entities,
        relationships=relationships,
        text_units=text_unit_rows,
        summary=summary,
    )


def _build_description_list_prompt(
    descriptions: Sequence[str],
    max_descriptions: int,
    max_description_chars: int,
) -> str:
    items: list[str] = []
    for description in descriptions:
        normalized = _normalize_whitespace(description)
        if not normalized:
            continue
        if len(normalized) > max_description_chars:
            truncated = normalized[:max_description_chars]
            if " " in truncated:
                truncated = truncated.rsplit(" ", 1)[0]
            normalized = f"{truncated} ..."
        items.append(normalized)

    if not items:
        return "- No description provided"

    limited = items[:max_descriptions]
    lines = [f"- {item}" for item in limited]
    if len(items) > len(limited):
        lines.append(
            f"- ... ({len(items) - len(limited)} additional descriptions omitted)"
        )
    return "\n".join(lines)


def _summary_cache_key(
    kind: str,
    company: str,
    label: str,
    descriptions: Sequence[str],
    max_length: int,
) -> str:
    digest = hashlib.sha1("\n".join(descriptions).encode("utf-8")).hexdigest()
    return f"{kind}|{company}|{label}|{max_length}|{digest}"


def _clean_summary_output(value: str) -> str:
    normalized = value.strip().strip('"').strip("'")
    if normalized.lower().startswith("output:"):
        normalized = normalized.split(":", 1)[1].strip()
    return _normalize_whitespace(normalized)


def _summarize_descriptions(
    kind: str,
    company: str,
    label: str,
    descriptions: Sequence[str],
    summarizer: "MistralService",
    cache: dict[str, str],
    max_length_words: int,
    max_new_tokens: int,
    temperature: float,
    max_descriptions_per_summary: int,
    max_description_chars: int,
) -> tuple[str, bool]:
    if not descriptions:
        return "", False
    if len(descriptions) == 1:
        return descriptions[0], False

    key = _summary_cache_key(
        kind=kind,
        company=company,
        label=label,
        descriptions=descriptions,
        max_length=max_length_words,
    )
    cached = cache.get(key)
    if cached:
        return cached, True

    description_list = _build_description_list_prompt(
        descriptions,
        max_descriptions=max_descriptions_per_summary,
        max_description_chars=max_description_chars,
    )
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        max_length=max_length_words,
        entity_name=label,
        description_list=description_list,
    )
    summary = summarizer.generate_answer(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    clean_summary = _clean_summary_output(summary)
    if clean_summary:
        cache[key] = clean_summary
        return clean_summary, False

    fallback = descriptions[0]
    cache[key] = fallback
    return fallback, False


def _load_summary_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    cache: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str):
            cache[key] = value
    return cache


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def _write_summary_cache(path: Path, cache: dict[str, str]) -> None:
    _write_json(path, cache)


def _apply_summaries(
    artifacts: CompanyArtifacts,
    summarizer: "MistralService",
    cache: dict[str, str],
    cache_path: Path,
    checkpoint_every: int,
    max_length_words: int,
    max_new_tokens: int,
    temperature: float,
    max_descriptions_per_summary: int,
    max_description_chars: int,
    verbose: bool,
) -> dict[str, int]:
    model_calls = 0
    cache_hits = 0
    failures = 0
    updates_since_checkpoint = 0

    for entity in artifacts.entities:
        descriptions = list(entity.get("description_list", []))
        if not descriptions:
            continue

        label = f"{entity['entity_name']}"
        try:
            summary, from_cache = _summarize_descriptions(
                kind="entity",
                company=artifacts.company,
                label=label,
                descriptions=descriptions,
                summarizer=summarizer,
                cache=cache,
                max_length_words=max_length_words,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_descriptions_per_summary=max_descriptions_per_summary,
                max_description_chars=max_description_chars,
            )
            entity["entity_description"] = summary or descriptions[0]
            if len(descriptions) > 1:
                if from_cache:
                    cache_hits += 1
                else:
                    model_calls += 1
                    updates_since_checkpoint += 1
        except Exception:
            failures += 1
            entity["entity_description"] = descriptions[0]

        if checkpoint_every > 0 and updates_since_checkpoint >= checkpoint_every:
            _write_summary_cache(cache_path, cache)
            updates_since_checkpoint = 0

    for relationship in artifacts.relationships:
        descriptions = list(relationship.get("description_list", []))
        if not descriptions:
            continue

        label = f"{relationship['source_entity']} | {relationship['target_entity']}"
        try:
            summary, from_cache = _summarize_descriptions(
                kind="relationship",
                company=artifacts.company,
                label=label,
                descriptions=descriptions,
                summarizer=summarizer,
                cache=cache,
                max_length_words=max_length_words,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_descriptions_per_summary=max_descriptions_per_summary,
                max_description_chars=max_description_chars,
            )
            relationship["relationship_description"] = summary or descriptions[0]
            if len(descriptions) > 1:
                if from_cache:
                    cache_hits += 1
                else:
                    model_calls += 1
                    updates_since_checkpoint += 1
        except Exception:
            failures += 1
            relationship["relationship_description"] = descriptions[0]

        if checkpoint_every > 0 and updates_since_checkpoint >= checkpoint_every:
            _write_summary_cache(cache_path, cache)
            updates_since_checkpoint = 0

    if updates_since_checkpoint > 0:
        _write_summary_cache(cache_path, cache)

    if verbose:
        print(
            f"[{artifacts.company}] summarization: model_calls={model_calls}, "
            f"cache_hits={cache_hits}, failures={failures}"
        )

    return {
        "summary_model_calls": model_calls,
        "summary_cache_hits": cache_hits,
        "summary_failures": failures,
    }


def _batched(
    items: Sequence[dict[str, Any]], batch_size: int
) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(items), batch_size):
        yield list(items[index : index + batch_size])


def _create_graph_schema(driver: Driver, database: str) -> None:
    create_company_constraint = (
        "CREATE CONSTRAINT company_name_unique IF NOT EXISTS "
        "FOR (company:Company) REQUIRE company.name IS UNIQUE"
    )
    create_entity_constraint = (
        "CREATE CONSTRAINT entity_company_name_unique IF NOT EXISTS "
        "FOR (entity:Entity) REQUIRE (entity.company, entity.name) IS UNIQUE"
    )

    with driver.session(database=database) as session:
        session.run(create_company_constraint).consume()
        session.run(create_entity_constraint).consume()


def _delete_company_graph(
    driver: Driver, database: str, companies: Sequence[str]
) -> None:
    if not companies:
        return

    query = """
    UNWIND $companies AS company
    MATCH (entity:Entity {company: company})
    DETACH DELETE entity
    """
    with driver.session(database=database) as session:
        session.run(query, companies=list(companies)).consume()


def _upsert_entities(
    driver: Driver,
    database: str,
    rows: Sequence[dict[str, Any]],
    batch_size: int,
    verbose: bool,
) -> None:
    query = """
    UNWIND $rows AS row
    MERGE (company:Company {name: row.company})
    MERGE (entity:Entity {company: row.company, name: row.entity_name})
    SET entity.type = row.entity_type,
        entity.alternate_types = row.alternate_types,
        entity.description = row.entity_description,
        entity.description_list = row.description_list,
        entity.frequency = row.frequency,
        entity.degree = row.degree,
        entity.text_unit_ids = row.text_unit_ids,
        entity.source_refs = row.source_refs,
        entity.source_count = row.source_count
    MERGE (company)-[:HAS_ENTITY]->(entity)
    """

    with driver.session(database=database) as session:
        for batch_index, batch in enumerate(_batched(rows, batch_size), start=1):
            session.run(query, rows=batch).consume()
            if verbose:
                print(f"Inserted entity batch {batch_index} ({len(batch)} rows)")


def _upsert_relationships(
    driver: Driver,
    database: str,
    rows: Sequence[dict[str, Any]],
    batch_size: int,
    verbose: bool,
) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (source:Entity {company: row.company, name: row.source_entity})
    MATCH (target:Entity {company: row.company, name: row.target_entity})
    MERGE (source)-[rel:RELATED_TO {company: row.company}]->(target)
    SET rel.source_type = row.source_entity_type,
        rel.target_type = row.target_entity_type,
        rel.description = row.relationship_description,
        rel.description_list = row.description_list,
        rel.weight = row.relationship_weight,
        rel.strength_max = row.relationship_strength_max,
        rel.occurrence_count = row.occurrence_count,
        rel.combined_degree = row.combined_degree,
        rel.text_unit_ids = row.text_unit_ids,
        rel.source_refs = row.source_refs,
        rel.source_count = row.source_count
    """

    with driver.session(database=database) as session:
        for batch_index, batch in enumerate(_batched(rows, batch_size), start=1):
            session.run(query, rows=batch).consume()
            if verbose:
                print(f"Inserted relationship batch {batch_index} ({len(batch)} rows)")


def _load_company_artifacts_for_neo4j(
    input_dir: Path,
    companies: Sequence[str] | None,
) -> list[CompanyArtifacts]:
    requested = {
        company.strip().lower() for company in companies or [] if company.strip()
    }
    artifacts: list[CompanyArtifacts] = []

    for company_dir in sorted(item for item in input_dir.iterdir() if item.is_dir()):
        company = company_dir.name.lower()
        if requested and company not in requested:
            continue

        entities_path = company_dir / "entities_summarized.json"
        relationships_path = company_dir / "relationships_summarized.json"
        text_units_path = company_dir / "text_units.json"
        summary_path = company_dir / "run_summary.json"

        if not entities_path.exists() or not relationships_path.exists():
            continue

        entities_payload = json.loads(entities_path.read_text(encoding="utf-8"))
        relationships_payload = json.loads(
            relationships_path.read_text(encoding="utf-8")
        )
        text_units_payload = (
            json.loads(text_units_path.read_text(encoding="utf-8"))
            if text_units_path.exists()
            else []
        )
        summary_payload = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else {"company": company}
        )

        if not isinstance(entities_payload, list) or not isinstance(
            relationships_payload, list
        ):
            continue

        artifacts.append(
            CompanyArtifacts(
                company=company,
                source_file=str(summary_payload.get("source_file", f"{company}.md")),
                entities=entities_payload,
                relationships=relationships_payload,
                text_units=text_units_payload
                if isinstance(text_units_payload, list)
                else [],
                summary=summary_payload
                if isinstance(summary_payload, dict)
                else {"company": company},
            )
        )

    return artifacts


def build_graph(args: argparse.Namespace) -> None:
    started = perf_counter()
    extraction_dir = Path(args.graph_extraction_dir)
    output_dir = Path(args.output_dir)

    if not extraction_dir.exists():
        raise FileNotFoundError(
            f"Graph extraction directory does not exist: {extraction_dir}"
        )

    companies = _list_companies(extraction_dir, args.companies)
    if not companies:
        raise RuntimeError(f"No companies selected in {extraction_dir}")

    artifacts_by_company: list[CompanyArtifacts] = []
    for company in companies:
        company_rows = _load_jsonl_rows(extraction_dir / company / "records_raw.jsonl")
        if not company_rows:
            continue

        artifacts = _aggregate_company(company=company, rows=company_rows)
        artifacts_by_company.append(artifacts)

        if args.verbose:
            print(
                f"[{company}] aggregated entities={len(artifacts.entities)} "
                f"relationships={len(artifacts.relationships)} text_units={len(artifacts.text_units)}"
            )

    if not artifacts_by_company:
        raise RuntimeError(
            "No company records were aggregated. Check records_raw.jsonl files."
        )

    cache_path = output_dir / "summary_cache.json"
    cache = _load_summary_cache(cache_path)

    summary_candidates_by_company: dict[str, int] = {
        artifacts.company: 0 for artifacts in artifacts_by_company
    }
    if args.summarize_descriptions:
        for artifacts in artifacts_by_company:
            company_candidates = sum(
                1
                for entity in artifacts.entities
                if len(entity.get("description_list", [])) > 1
            )
            company_candidates += sum(
                1
                for relationship in artifacts.relationships
                if len(relationship.get("description_list", [])) > 1
            )
            summary_candidates_by_company[artifacts.company] = company_candidates

    summary_call_candidates = sum(summary_candidates_by_company.values())

    summarization_stats_by_company: dict[str, dict[str, int]] = {
        artifacts.company: {
            "summary_model_calls": 0,
            "summary_cache_hits": 0,
            "summary_failures": 0,
        }
        for artifacts in artifacts_by_company
    }

    if args.summarize_descriptions and summary_call_candidates > 0:
        if args.verbose:
            print(
                f"Summarization groups requiring model/cached resolution: {summary_call_candidates}"
            )

        summarizer = _create_mistral_service(
            model_dir=Path(args.model_dir),
            default_max_new_tokens=args.summary_max_new_tokens,
            default_temperature=args.summary_temperature,
        )

        for artifacts in artifacts_by_company:
            run_stats = _apply_summaries(
                artifacts=artifacts,
                summarizer=summarizer,
                cache=cache,
                cache_path=cache_path,
                checkpoint_every=args.summary_checkpoint_every,
                max_length_words=args.summary_max_length_words,
                max_new_tokens=args.summary_max_new_tokens,
                temperature=args.summary_temperature,
                max_descriptions_per_summary=args.max_descriptions_per_summary,
                max_description_chars=args.max_description_chars,
                verbose=args.verbose,
            )
            summarization_stats_by_company[artifacts.company] = run_stats

        _write_summary_cache(cache_path, cache)

    summary_model_calls = sum(
        stats["summary_model_calls"]
        for stats in summarization_stats_by_company.values()
    )
    summary_cache_hits = sum(
        stats["summary_cache_hits"] for stats in summarization_stats_by_company.values()
    )
    summary_failures = sum(
        stats["summary_failures"] for stats in summarization_stats_by_company.values()
    )

    for artifacts in artifacts_by_company:
        company_output_dir = output_dir / artifacts.company
        company_output_dir.mkdir(parents=True, exist_ok=True)

        company_stats = summarization_stats_by_company.get(
            artifacts.company,
            {
                "summary_model_calls": 0,
                "summary_cache_hits": 0,
                "summary_failures": 0,
            },
        )

        summary_payload = {
            **artifacts.summary,
            "summarize_descriptions": bool(args.summarize_descriptions),
            "summary_model_calls": company_stats["summary_model_calls"],
            "summary_cache_hits": company_stats["summary_cache_hits"],
            "summary_failures": company_stats["summary_failures"],
            "summary_call_candidates": summary_candidates_by_company.get(
                artifacts.company, 0
            ),
            "output_dir": str(company_output_dir),
        }

        _write_json(company_output_dir / "entities_summarized.json", artifacts.entities)
        _write_json(
            company_output_dir / "relationships_summarized.json",
            artifacts.relationships,
        )
        _write_json(company_output_dir / "text_units.json", artifacts.text_units)
        _write_json(
            company_output_dir / "graph_records.json",
            {
                "company": artifacts.company,
                "source_file": artifacts.source_file,
                "entities": artifacts.entities,
                "relationships": artifacts.relationships,
            },
        )
        _write_json(company_output_dir / "run_summary.json", summary_payload)

    aggregate_summary = {
        "companies": [artifacts.company for artifacts in artifacts_by_company],
        "companies_count": len(artifacts_by_company),
        "entities_count": sum(
            len(artifacts.entities) for artifacts in artifacts_by_company
        ),
        "relationships_count": sum(
            len(artifacts.relationships) for artifacts in artifacts_by_company
        ),
        "text_units_count": sum(
            len(artifacts.text_units) for artifacts in artifacts_by_company
        ),
        "summarize_descriptions": bool(args.summarize_descriptions),
        "summary_call_candidates": summary_call_candidates,
        "summary_model_calls": summary_model_calls,
        "summary_cache_hits": summary_cache_hits,
        "summary_failures": summary_failures,
        "elapsed_seconds": round(perf_counter() - started, 2),
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "run_summary.json", aggregate_summary)

    print(
        "Built graph artifacts for "
        f"{len(artifacts_by_company)} companies -> "
        f"entities={aggregate_summary['entities_count']}, "
        f"relationships={aggregate_summary['relationships_count']}"
    )


def load_graph(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Graph index input directory does not exist: {input_dir}"
        )

    artifacts = _load_company_artifacts_for_neo4j(input_dir, args.companies)
    if not artifacts:
        raise RuntimeError(f"No summarized graph artifacts found in {input_dir}")

    companies = sorted({item.company for item in artifacts})
    entity_rows = [entity for item in artifacts for entity in item.entities]
    relationship_rows = [
        relationship for item in artifacts for relationship in item.relationships
    ]

    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    )
    try:
        driver.verify_connectivity()
        _create_graph_schema(driver, args.neo4j_database)

        if args.replace_existing:
            if args.verbose:
                print(f"Replacing existing Entity graph for: {', '.join(companies)}")
            _delete_company_graph(driver, args.neo4j_database, companies)

        _upsert_entities(
            driver,
            args.neo4j_database,
            rows=entity_rows,
            batch_size=args.db_batch_size,
            verbose=args.verbose,
        )
        _upsert_relationships(
            driver,
            args.neo4j_database,
            rows=relationship_rows,
            batch_size=args.db_batch_size,
            verbose=args.verbose,
        )
    finally:
        driver.close()

    print(
        "Loaded summarized graph into Neo4j -> "
        f"companies={len(companies)}, entities={len(entity_rows)}, "
        f"relationships={len(relationship_rows)}"
    )


def build_and_load_graph(args: argparse.Namespace) -> None:
    build_graph(args)

    load_args = argparse.Namespace(
        input_dir=args.output_dir,
        companies=args.companies,
        replace_existing=args.replace_existing,
        db_batch_size=args.db_batch_size,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        verbose=args.verbose,
    )
    load_graph(load_args)


def _add_neo4j_arguments(parser: argparse.ArgumentParser) -> None:
    default_user, default_password = _resolve_neo4j_auth()
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j Bolt URI.",
    )
    parser.add_argument("--neo4j-user", default=default_user, help="Neo4j username.")
    parser.add_argument(
        "--neo4j-password",
        default=default_password,
        help="Neo4j password.",
    )
    parser.add_argument(
        "--neo4j-database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build company-scoped, summarized entity/relationship graph artifacts "
            "from extraction outputs and load them into Neo4j."
        )
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser(
        "build", help="Aggregate and summarize graph extraction outputs"
    )
    build_cmd.add_argument(
        "--graph-extraction-dir",
        default=str(DEFAULT_GRAPH_EXTRACTION_DIR),
        help="Directory containing per-company records_raw.jsonl outputs.",
    )
    build_cmd.add_argument(
        "--output-dir",
        default=str(DEFAULT_GRAPH_INDEX_DIR),
        help="Directory where summarized graph artifacts are written.",
    )
    build_cmd.add_argument(
        "--companies",
        nargs="+",
        default=None,
        help="Optional company list (markdown stems) to process.",
    )
    build_cmd.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to local Mistral model directory for description summarization.",
    )
    build_cmd.add_argument(
        "--summary-max-new-tokens",
        type=int,
        default=DEFAULT_SUMMARY_MAX_NEW_TOKENS,
        help="Max new tokens for each summary generation call.",
    )
    build_cmd.add_argument(
        "--summary-temperature",
        type=float,
        default=DEFAULT_SUMMARY_TEMPERATURE,
        help="Sampling temperature for summary generation.",
    )
    build_cmd.add_argument(
        "--summary-max-length-words",
        type=int,
        default=DEFAULT_SUMMARY_MAX_LENGTH_WORDS,
        help="Word limit requested in the summarization prompt.",
    )
    build_cmd.add_argument(
        "--max-descriptions-per-summary",
        type=int,
        default=20,
        help="Maximum descriptions included in one summarization prompt.",
    )
    build_cmd.add_argument(
        "--max-description-chars",
        type=int,
        default=420,
        help="Maximum characters kept per description in summarization prompts.",
    )
    build_cmd.add_argument(
        "--summary-checkpoint-every",
        type=int,
        default=DEFAULT_SUMMARY_CHECKPOINT_EVERY,
        help="Persist summary cache every N new model summaries (0 disables checkpointing).",
    )
    build_cmd.add_argument(
        "--no-summarize-descriptions",
        dest="summarize_descriptions",
        action="store_false",
        help="Skip LLM summarization and keep longest raw description per item.",
    )
    build_cmd.set_defaults(summarize_descriptions=True)
    build_cmd.set_defaults(handler=build_graph)

    load_cmd = subparsers.add_parser(
        "load", help="Load summarized graph artifacts into Neo4j"
    )
    load_cmd.add_argument(
        "--input-dir",
        default=str(DEFAULT_GRAPH_INDEX_DIR),
        help="Directory containing per-company summarized graph artifacts.",
    )
    load_cmd.add_argument(
        "--companies",
        nargs="+",
        default=None,
        help="Optional company list (directory names) to load.",
    )
    load_cmd.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete existing Entity graph nodes for selected companies before insert.",
    )
    load_cmd.add_argument(
        "--db-batch-size",
        type=int,
        default=200,
        help="Batch size for writing entities and relationships to Neo4j.",
    )
    _add_neo4j_arguments(load_cmd)
    load_cmd.set_defaults(handler=load_graph)

    build_load_cmd = subparsers.add_parser(
        "build-load",
        help="Run build and then load into Neo4j",
    )
    build_load_cmd.add_argument(
        "--graph-extraction-dir",
        default=str(DEFAULT_GRAPH_EXTRACTION_DIR),
        help="Directory containing per-company records_raw.jsonl outputs.",
    )
    build_load_cmd.add_argument(
        "--output-dir",
        default=str(DEFAULT_GRAPH_INDEX_DIR),
        help="Directory where summarized graph artifacts are written.",
    )
    build_load_cmd.add_argument(
        "--companies",
        nargs="+",
        default=None,
        help="Optional company list (markdown stems) to process and load.",
    )
    build_load_cmd.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to local Mistral model directory for description summarization.",
    )
    build_load_cmd.add_argument(
        "--summary-max-new-tokens",
        type=int,
        default=DEFAULT_SUMMARY_MAX_NEW_TOKENS,
        help="Max new tokens for each summary generation call.",
    )
    build_load_cmd.add_argument(
        "--summary-temperature",
        type=float,
        default=DEFAULT_SUMMARY_TEMPERATURE,
        help="Sampling temperature for summary generation.",
    )
    build_load_cmd.add_argument(
        "--summary-max-length-words",
        type=int,
        default=DEFAULT_SUMMARY_MAX_LENGTH_WORDS,
        help="Word limit requested in the summarization prompt.",
    )
    build_load_cmd.add_argument(
        "--max-descriptions-per-summary",
        type=int,
        default=20,
        help="Maximum descriptions included in one summarization prompt.",
    )
    build_load_cmd.add_argument(
        "--max-description-chars",
        type=int,
        default=420,
        help="Maximum characters kept per description in summarization prompts.",
    )
    build_load_cmd.add_argument(
        "--summary-checkpoint-every",
        type=int,
        default=DEFAULT_SUMMARY_CHECKPOINT_EVERY,
        help="Persist summary cache every N new model summaries (0 disables checkpointing).",
    )
    build_load_cmd.add_argument(
        "--no-summarize-descriptions",
        dest="summarize_descriptions",
        action="store_false",
        help="Skip LLM summarization and keep longest raw description per item.",
    )
    build_load_cmd.set_defaults(summarize_descriptions=True)
    build_load_cmd.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete existing Entity graph nodes for selected companies before insert.",
    )
    build_load_cmd.add_argument(
        "--db-batch-size",
        type=int,
        default=200,
        help="Batch size for writing entities and relationships to Neo4j.",
    )
    _add_neo4j_arguments(build_load_cmd)
    build_load_cmd.set_defaults(handler=build_and_load_graph)

    for command in (build_cmd, load_cmd, build_load_cmd):
        command.add_argument(
            "--verbose",
            action="store_true",
            help="Print detailed progress logs.",
        )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"build", "build-load"}:
        if args.summary_max_new_tokens < 16:
            parser.error("--summary-max-new-tokens must be >= 16")
        if args.summary_max_length_words < 20:
            parser.error("--summary-max-length-words must be >= 20")
        if args.max_descriptions_per_summary < 1:
            parser.error("--max-descriptions-per-summary must be >= 1")
        if args.max_description_chars < 80:
            parser.error("--max-description-chars must be >= 80")
        if args.summary_checkpoint_every < 0:
            parser.error("--summary-checkpoint-every must be >= 0")

    if hasattr(args, "db_batch_size") and args.db_batch_size < 1:
        parser.error("--db-batch-size must be >= 1")

    args.handler(args)


if __name__ == "__main__":
    main()
