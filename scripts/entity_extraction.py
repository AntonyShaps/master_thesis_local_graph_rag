from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENTITY_TYPES
from paths import DATA_MARKDOWN_DIR, MODELS_DIR

if TYPE_CHECKING:
    from app.mistral_service import MistralService


GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a sustainability- or ESG-related text document and a list of entity types, identify all relevant entities of those types and all clearly supported relationships among them.

The text may describe:
- corporate sustainability strategy
- climate, energy, water, biodiversity, or supply-chain topics
- AI, data centers, hardware, products, or technical infrastructure
- goals, targets, metrics, standards, partnerships, projects, and risks

Your job is to build a high-quality knowledge graph for ESG analysis.

-Steps-

1. Identify all entities.
For each identified entity, extract the following information:
- entity_name: Canonical name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: A comprehensive description of the entity based only on the provided text

Format each entity as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are clearly related.
For each related pair, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: a concise explanation of how the two entities are related, based only on the provided text
- relationship_strength: a numeric score from 1 to 10 indicating the strength and explicitness of the relationship in the text

Format each relationship as:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all entities and relationships identified in steps 1 and 2.
Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

-Entity Extraction Rules-
- Extract only entities that are explicitly stated or clearly implied by the text.
- Use the most specific canonical name available in the text.
- Normalize entity names:
  - Convert entity names to uppercase.
  - Remove unnecessary punctuation unless part of the official name.
  - Use a single canonical form for repeated mentions (for example, "GOOGLE" instead of both "Google" and "the company").
- Do not create duplicate entities.
- Do not create generic entities such as "COMPANY", "REPORT", "TEAM", or "INDUSTRY" unless they are clearly named and meaningful in context.
- If a metric or target is described with a number, keep the quantitative detail inside the description rather than creating separate number-only entities.
- Prefer precision over recall: do not guess.

-Entity Type Guidance-
Use the entity types exactly as provided in [{entity_types}].

Interpret them as follows:
- COMPANY: report issuer or major corporate actor
- ORGANIZATION: partner, supplier, NGO, coalition, research institute, utility, regulator, industry body, or other institution
- FACILITY_ASSET: physical site or infrastructure asset such as data center, campus, office, manufacturing site, plant, grid asset, warehouse, or server fleet
- GEO: country, region, city, state, watershed, grid region, or other geographic area
- ESG_TOPIC: sustainability theme or material topic such as climate, water stewardship, biodiversity, circularity, human rights, or responsible sourcing
- GOAL_TARGET: formal target, ambition, commitment, moonshot, or deadline-based sustainability objective
- METRIC: named KPI or measured indicator such as Scope 1 emissions, PUE, CFE percentage, renewable electricity match, water restored, or energy efficiency
- PRODUCT_TECHNOLOGY: named product, model, chip, platform, API, tool, software system, or technical solution
- INITIATIVE_PROGRAM: ongoing internal or external program, engagement mechanism, procurement mechanism, framework, or structured effort
- PROJECT: specific deployment, named implementation, restoration effort, infrastructure project, or named energy project
- STANDARD_FRAMEWORK: reporting, audit, certification, or target-setting framework such as GRI, SASB, TCFD, SBTi, LEED, ISO 14001, ISO 50001
- RESOURCE: energy source, material, water, waste stream, or operational input such as solar, wind, geothermal, freshwater, steel, concrete, semiconductors
- PERSON: named executive, leader, or stakeholder
- RISK_CHALLENGE: clearly identified risk, barrier, dependency, uncertainty, or challenge

-Relationship Extraction Rules-
Extract only relationships that are supported by the text.
Common relationship patterns include:
- COMPANY sets GOAL_TARGET
- COMPANY reports METRIC
- COMPANY operates or uses FACILITY_ASSET
- COMPANY partners with ORGANIZATION
- COMPANY develops or deploys PRODUCT_TECHNOLOGY
- COMPANY runs INITIATIVE_PROGRAM
- COMPANY supports PROJECT
- COMPANY follows or reports against STANDARD_FRAMEWORK
- FACILITY_ASSET located in GEO
- PROJECT located in GEO
- PROJECT uses RESOURCE
- PRODUCT_TECHNOLOGY improves or supports ESG_TOPIC
- METRIC measures ESG_TOPIC, FACILITY_ASSET, PRODUCT_TECHNOLOGY, or COMPANY performance
- GOAL_TARGET relates to ESG_TOPIC or METRIC
- ORGANIZATION participates in INITIATIVE_PROGRAM or PROJECT
- RISK_CHALLENGE affects COMPANY, FACILITY_ASSET, PROJECT, ESG_TOPIC, or GOAL_TARGET

-Relationship Strength Guidance-
Use:
- 9-10 when the relationship is direct and explicit
- 6-8 when the relationship is clearly supported but slightly indirect
- 3-5 when the relationship is weaker but still reasonable and text-grounded
- 1-2 only for very weak but still defensible relationships

-Important Constraints-
- Do not use outside knowledge.
- Do not infer beyond what the text supports.
- Do not output explanations, headings, or commentary outside the required list format.
- Do not output any entity or relationship twice.
- If the same real-world entity appears in multiple roles, assign the single best-fitting type based on the text.
- If uncertain between INITIATIVE_PROGRAM and PROJECT:
  - use INITIATIVE_PROGRAM for ongoing structured efforts, frameworks, or engagement mechanisms
  - use PROJECT for specific named implementations, sites, or deployments
- If uncertain between METRIC and GOAL_TARGET:
  - use METRIC for what is measured
  - use GOAL_TARGET for what is intended or committed

######################
-Examples-
######################
Example 1:
Entity_types: COMPANY,GOAL_TARGET,METRIC,ESG_TOPIC
Text:
The company aims to achieve net zero emissions across its value chain by 2030. In 2024, its Scope 1 and 2 emissions decreased by 12% compared to the previous year.
######################
Output:
("entity"{tuple_delimiter}THE COMPANY{tuple_delimiter}COMPANY{tuple_delimiter}A company that reports sustainability performance and sets emissions-related goals)
{record_delimiter}
("entity"{tuple_delimiter}NET ZERO EMISSIONS ACROSS ITS VALUE CHAIN BY 2030{tuple_delimiter}GOAL_TARGET{tuple_delimiter}A formal company goal to achieve net zero emissions across its value chain by 2030)
{record_delimiter}
("entity"{tuple_delimiter}SCOPE 1 AND 2 EMISSIONS{tuple_delimiter}METRIC{tuple_delimiter}A greenhouse gas emissions metric reported by the company, which decreased by 12% in 2024 compared to the previous year)
{record_delimiter}
("entity"{tuple_delimiter}EMISSIONS{tuple_delimiter}ESG_TOPIC{tuple_delimiter}An ESG topic concerning greenhouse gas emissions and decarbonization)
{record_delimiter}
("relationship"{tuple_delimiter}THE COMPANY{tuple_delimiter}NET ZERO EMISSIONS ACROSS ITS VALUE CHAIN BY 2030{tuple_delimiter}The company set a formal target to achieve net zero emissions across its value chain by 2030{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}THE COMPANY{tuple_delimiter}SCOPE 1 AND 2 EMISSIONS{tuple_delimiter}The company reported Scope 1 and 2 emissions performance for 2024{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}NET ZERO EMISSIONS ACROSS ITS VALUE CHAIN BY 2030{tuple_delimiter}EMISSIONS{tuple_delimiter}The target is specifically about reducing emissions{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}SCOPE 1 AND 2 EMISSIONS{tuple_delimiter}EMISSIONS{tuple_delimiter}Scope 1 and 2 emissions are a metric related to the emissions topic{tuple_delimiter}9)
{completion_delimiter}

######################
Example 2:
Entity_types: COMPANY,ORGANIZATION,PROJECT,RESOURCE,GEO
Text:
Google signed an agreement with Kairos Power to support small modular nuclear reactors in the United States.
######################
Output:
("entity"{tuple_delimiter}GOOGLE{tuple_delimiter}COMPANY{tuple_delimiter}A company participating in an agreement to support advanced nuclear energy deployment)
{record_delimiter}
("entity"{tuple_delimiter}KAIROS POWER{tuple_delimiter}ORGANIZATION{tuple_delimiter}An organization developing small modular nuclear reactor technology)
{record_delimiter}
("entity"{tuple_delimiter}SMALL MODULAR NUCLEAR REACTORS{tuple_delimiter}PROJECT{tuple_delimiter}A nuclear energy deployment effort involving small modular reactors supported through an agreement)
{record_delimiter}
("entity"{tuple_delimiter}NUCLEAR ENERGY{tuple_delimiter}RESOURCE{tuple_delimiter}An energy resource associated with small modular reactor technology)
{record_delimiter}
("entity"{tuple_delimiter}UNITED STATES{tuple_delimiter}GEO{tuple_delimiter}The country in which the small modular nuclear reactor effort is supported)
{record_delimiter}
("relationship"{tuple_delimiter}GOOGLE{tuple_delimiter}KAIROS POWER{tuple_delimiter}Google signed an agreement with Kairos Power{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}GOOGLE{tuple_delimiter}SMALL MODULAR NUCLEAR REACTORS{tuple_delimiter}Google supports the deployment of small modular nuclear reactors{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}KAIROS POWER{tuple_delimiter}SMALL MODULAR NUCLEAR REACTORS{tuple_delimiter}Kairos Power is the organization developing the reactor project{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}SMALL MODULAR NUCLEAR REACTORS{tuple_delimiter}NUCLEAR ENERGY{tuple_delimiter}The reactor project is based on nuclear energy{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}SMALL MODULAR NUCLEAR REACTORS{tuple_delimiter}UNITED STATES{tuple_delimiter}The project is supported in the United States{tuple_delimiter}8)
{completion_delimiter}

######################
Example 3:
Entity_types: COMPANY,FACILITY_ASSET,GEO,METRIC,RESOURCE
Text:
Meta data centers in high water stress regions use water budgeting and flow meter audits to reduce water use.
######################
Output:
("entity"{tuple_delimiter}META{tuple_delimiter}COMPANY{tuple_delimiter}A company applying water reduction practices in its data center operations)
{record_delimiter}
("entity"{tuple_delimiter}DATA CENTERS{tuple_delimiter}FACILITY_ASSET{tuple_delimiter}Operational facilities used by Meta where water reduction practices are applied)
{record_delimiter}
("entity"{tuple_delimiter}HIGH WATER STRESS REGIONS{tuple_delimiter}GEO{tuple_delimiter}Geographic regions characterized by high water stress where Meta applies water management practices)
{record_delimiter}
("entity"{tuple_delimiter}WATER USE{tuple_delimiter}METRIC{tuple_delimiter}A metric relating to operational water consumption that Meta is seeking to reduce)
{record_delimiter}
("entity"{tuple_delimiter}WATER{tuple_delimiter}RESOURCE{tuple_delimiter}A resource used in data center operations and managed through reduction practices)
{record_delimiter}
("relationship"{tuple_delimiter}META{tuple_delimiter}DATA CENTERS{tuple_delimiter}Meta operates data centers where water management practices are applied{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}DATA CENTERS{tuple_delimiter}HIGH WATER STRESS REGIONS{tuple_delimiter}The data centers discussed are located in high water stress regions{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}META{tuple_delimiter}WATER USE{tuple_delimiter}Meta is working to reduce operational water use{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}WATER USE{tuple_delimiter}WATER{tuple_delimiter}Water use is the metric associated with the water resource{tuple_delimiter}9)
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_MAX_CHARS_PER_UNIT = 3200
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("ENTITY_LLM_MAX_NEW_TOKENS", "1400"))
DEFAULT_TEMPERATURE = float(os.getenv("ENTITY_LLM_TEMPERATURE", "0.0"))
DEFAULT_MODEL_DIR = Path(
    os.getenv("MISTRAL_MODEL_DIR", str(MODELS_DIR / "Mistral-7B-Instruct-v0.3"))
)
DEFAULT_OUTPUT_DIR = DATA_MARKDOWN_DIR.parent / "graph_extraction"

PAGE_BLOCK_RE = re.compile(
    r"--- PAGE\s+([0-9]+)\s+START ---\s*(.*?)\s*--- PAGE\s+\1\s+END ---",
    re.IGNORECASE | re.DOTALL,
)

IMAGE_TAG_RE = re.compile(r"<img>.*?</img>", re.IGNORECASE | re.DOTALL)
GENERIC_TAG_RE = re.compile(r"</?[^>]+>")

GENERIC_COMPANY_REFERENCES = {
    "THE COMPANY",
    "OUR COMPANY",
    "THE GROUP",
    "THE BUSINESS",
    "THE ISSUER",
    "WE",
    "OUR",
}

NAVIGATION_FOOTER_TOKENS = {
    "message from our ceo",
    "introduction",
    "age of ai",
    "energy, efficiency, and climate",
    "people, diversity, and inclusion",
    "product value chain",
    "responsible business",
    "sustainability indicators",
    "data center energy",
    "supply chain energy",
    "resource efficiency",
    "research",
    "disaster response",
    "protecting the planet",
    "appendix",
    "climate",
    "water",
    "biodiversity",
}

STRUCTURAL_NOISE_LINE_PATTERNS = [
    re.compile(r"^graph\s+[A-Za-z0-9_]+\s*$", re.IGNORECASE),
    re.compile(r"^subgraph\b", re.IGNORECASE),
    re.compile(r"^style\s+[A-Za-z0-9_]+\s+", re.IGNORECASE),
    re.compile(r"^[XY]_axis", re.IGNORECASE),
    re.compile(r"^Y_axis_tick_labels\[", re.IGNORECASE),
    re.compile(r"^Legend\s*$", re.IGNORECASE),
    re.compile(r"^[\[\](){}|\-_=*]+$"),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^[A-Za-z]\[[^\]]*\]$"),
]

ENTITY_NOISE_PATTERNS = [
    re.compile(r"^[XY]_AXIS", re.IGNORECASE),
    re.compile(r"TICK_LABEL", re.IGNORECASE),
    re.compile(r"^FIGURE\s+\d+", re.IGNORECASE),
    re.compile(r"^TABLE\s+OF\s+CONTENTS$", re.IGNORECASE),
    re.compile(r"^CONTENTS$", re.IGNORECASE),
]

FATAL_MODEL_ERROR_PATTERNS = [
    re.compile(r"cuda error", re.IGNORECASE),
    re.compile(r"cublas", re.IGNORECASE),
    re.compile(r"cudnn", re.IGNORECASE),
    re.compile(r"device-side assert", re.IGNORECASE),
    re.compile(r"illegal memory access", re.IGNORECASE),
    re.compile(r"unknown error", re.IGNORECASE),
]


@dataclass(frozen=True)
class PageSection:
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractionUnit:
    page_number: int
    unit_index: int
    text: str


_DEFAULT_MISTRAL_SERVICE: MistralService | None = None


def _create_mistral_service(
    model_dir: Path,
    default_max_new_tokens: int,
    default_temperature: float,
) -> MistralService:
    from app.mistral_service import MistralService as RuntimeMistralService

    return RuntimeMistralService(
        model_dir=model_dir,
        default_max_new_tokens=default_max_new_tokens,
        default_temperature=default_temperature,
    )


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _canonical_company_name(company_name: str | None) -> str | None:
    if not company_name:
        return None
    canonical = _normalize_whitespace(company_name).upper()
    return canonical or None


def _canonical_entity_name(name: str, company_name: str | None = None) -> str:
    cleaned = _normalize_whitespace(name).strip("`\"' ")
    cleaned = cleaned.strip(".,:;")
    canonical = cleaned.upper()

    company = _canonical_company_name(company_name)
    if company and canonical in GENERIC_COMPANY_REFERENCES:
        return company

    return canonical


def _is_navigation_noise_line(line: str) -> bool:
    if "|" not in line:
        return False

    parts = [part.strip().lower() for part in line.split("|") if part.strip()]
    if len(parts) < 4:
        return False

    token_hits = sum(1 for part in parts if part in NAVIGATION_FOOTER_TOKENS)
    return token_hits >= 3


def _is_structural_noise_line(line: str) -> bool:
    normalized = _normalize_whitespace(line)
    if not normalized:
        return False

    for pattern in STRUCTURAL_NOISE_LINE_PATTERNS:
        if pattern.search(normalized):
            return True

    return False


def _is_probably_noise_entity_name(entity_name: str, entity_type: str) -> bool:
    candidate = _normalize_whitespace(entity_name).upper()
    if not candidate:
        return True

    if not re.search(r"[A-Z]", candidate):
        return True

    if entity_type not in {"METRIC", "GOAL_TARGET"} and re.fullmatch(
        r"\d{4}", candidate
    ):
        return True

    for pattern in ENTITY_NOISE_PATTERNS:
        if pattern.search(candidate):
            return True

    return False


def _is_fatal_model_error(error_message: str) -> bool:
    normalized = _normalize_whitespace(error_message)
    if not normalized:
        return False

    return any(pattern.search(normalized) for pattern in FATAL_MODEL_ERROR_PATTERNS)


def _clean_page_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = IMAGE_TAG_RE.sub(" ", normalized)
    normalized = GENERIC_TAG_RE.sub(" ", normalized)

    cleaned_lines: list[str] = []
    previous_line = ""
    duplicate_count = 0

    for raw_line in normalized.split("\n"):
        line = _normalize_whitespace(raw_line)
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            previous_line = ""
            duplicate_count = 0
            continue

        if _is_navigation_noise_line(line) or _is_structural_noise_line(line):
            continue

        if line == previous_line:
            duplicate_count += 1
            if duplicate_count > 2:
                continue
        else:
            duplicate_count = 1

        cleaned_lines.append(line)
        previous_line = line

    collapsed = "\n".join(cleaned_lines)
    collapsed = re.sub(r"[ \t]+", " ", collapsed)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def _split_oversized_paragraph(paragraph: str, max_chars_per_unit: int) -> list[str]:
    words = paragraph.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0

    for word in words:
        add_len = len(word) + (1 if current_words else 0)
        if current_words and current_len + add_len > max_chars_per_unit:
            chunks.append(" ".join(current_words).strip())
            current_words = [word]
            current_len = len(word)
        else:
            current_words.append(word)
            current_len += add_len

    if current_words:
        chunks.append(" ".join(current_words).strip())

    return chunks


def split_text_units(
    text: str, max_chars_per_unit: int = DEFAULT_MAX_CHARS_PER_UNIT
) -> list[str]:
    if max_chars_per_unit < 500:
        raise ValueError("max_chars_per_unit must be >= 500")

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n{2,}", text)
        if paragraph.strip()
    ]
    if not paragraphs:
        return []

    units: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars_per_unit:
            if current_parts:
                units.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_len = 0

            units.extend(_split_oversized_paragraph(paragraph, max_chars_per_unit))
            continue

        add_len = len(paragraph) + (2 if current_parts else 0)
        if current_parts and current_len + add_len > max_chars_per_unit:
            units.append("\n\n".join(current_parts).strip())
            current_parts = [paragraph]
            current_len = len(paragraph)
        else:
            current_parts.append(paragraph)
            current_len += add_len

    if current_parts:
        units.append("\n\n".join(current_parts).strip())

    return units


def parse_markdown_pages(markdown_text: str) -> list[PageSection]:
    pages: list[PageSection] = []
    for page_number_raw, page_text in PAGE_BLOCK_RE.findall(markdown_text):
        page_number = int(page_number_raw)
        pages.append(PageSection(page_number=page_number, text=page_text.strip()))

    if not pages:
        fallback_text = markdown_text.strip()
        if fallback_text:
            pages.append(PageSection(page_number=1, text=fallback_text))

    return pages


def create_extraction_prompt(
    entity_types: Sequence[str],
    text: str,
    company_name: str | None = None,
    tuple_delimiter: str = DEFAULT_TUPLE_DELIMITER,
    record_delimiter: str = DEFAULT_RECORD_DELIMITER,
    completion_delimiter: str = DEFAULT_COMPLETION_DELIMITER,
) -> str:
    company = _canonical_company_name(company_name)
    if company:
        grounded_text = (
            f"SOURCE_REPORT_COMPANY: {company}\n"
            "If the text uses first-person references like 'we', 'our', "
            "or 'the company', resolve them to SOURCE_REPORT_COMPANY.\n\n"
            f"{text}"
        )
    else:
        grounded_text = text

    entity_types_string = ",".join(entity_types)
    return GRAPH_EXTRACTION_PROMPT.format(
        entity_types=entity_types_string,
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        completion_delimiter=completion_delimiter,
        input_text=grounded_text,
    )


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _normalize_strength(value: Any) -> int:
    if isinstance(value, (int, float)):
        score = int(round(float(value)))
    else:
        text_value = _normalize_whitespace(str(value))
        match = re.search(r"-?\d+(?:\.\d+)?", text_value)
        score = int(round(float(match.group(0)))) if match else 5

    return max(1, min(10, score))


def _parse_entity_tuple(record: str, tuple_delimiter: str) -> dict[str, Any] | None:
    delimiter = re.escape(tuple_delimiter)
    pattern = re.compile(
        rf'^\(\s*["\']entity["\']\s*{delimiter}\s*(.*?)\s*{delimiter}\s*(.*?)\s*{delimiter}\s*(.*)\)\s*$',
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.match(record.strip().rstrip(","))
    if not match:
        return None

    return {
        "record_type": "entity",
        "entity_name": _normalize_whitespace(match.group(1)),
        "entity_type": _normalize_whitespace(match.group(2)).upper(),
        "entity_description": _normalize_whitespace(match.group(3)),
    }


def _parse_relationship_tuple(
    record: str, tuple_delimiter: str
) -> dict[str, Any] | None:
    delimiter = re.escape(tuple_delimiter)
    pattern = re.compile(
        rf'^\(\s*["\']relationship["\']\s*{delimiter}\s*(.*?)\s*{delimiter}\s*(.*?)\s*{delimiter}\s*(.*?)\s*{delimiter}\s*(.*)\)\s*$',
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.match(record.strip().rstrip(","))
    if not match:
        return None

    return {
        "record_type": "relationship",
        "source_entity": _normalize_whitespace(match.group(1)),
        "target_entity": _normalize_whitespace(match.group(2)),
        "relationship_description": _normalize_whitespace(match.group(3)),
        "relationship_strength": _normalize_strength(match.group(4)),
    }


def _parse_json_records(text: str) -> list[dict[str, Any]] | None:
    cleaned = _strip_code_fence(text)

    candidates = [cleaned]
    list_start = cleaned.find("[")
    list_end = cleaned.rfind("]")
    if list_start != -1 and list_end > list_start:
        candidates.append(cleaned[list_start : list_end + 1])

    dict_start = cleaned.find("{")
    dict_end = cleaned.rfind("}")
    if dict_start != -1 and dict_end > dict_start:
        candidates.append(cleaned[dict_start : dict_end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        records: list[dict[str, Any]] = []

        def _append_record(item: dict[str, Any]) -> None:
            record_type = str(
                item.get("record_type") or item.get("type") or item.get("kind") or ""
            ).lower()

            if (
                record_type in {"entity", "node"}
                or "entity_name" in item
                and "entity_type" in item
            ):
                records.append(
                    {
                        "record_type": "entity",
                        "entity_name": _normalize_whitespace(
                            str(item.get("entity_name") or item.get("name") or "")
                        ),
                        "entity_type": _normalize_whitespace(
                            str(item.get("entity_type") or item.get("category") or "")
                        ).upper(),
                        "entity_description": _normalize_whitespace(
                            str(
                                item.get("entity_description")
                                or item.get("description")
                                or ""
                            )
                        ),
                    }
                )
                return

            if (
                record_type in {"relationship", "edge"}
                or "source_entity" in item
                and "target_entity" in item
            ):
                records.append(
                    {
                        "record_type": "relationship",
                        "source_entity": _normalize_whitespace(
                            str(item.get("source_entity") or item.get("source") or "")
                        ),
                        "target_entity": _normalize_whitespace(
                            str(item.get("target_entity") or item.get("target") or "")
                        ),
                        "relationship_description": _normalize_whitespace(
                            str(
                                item.get("relationship_description")
                                or item.get("description")
                                or ""
                            )
                        ),
                        "relationship_strength": _normalize_strength(
                            item.get("relationship_strength")
                            or item.get("strength")
                            or item.get("weight")
                            or 5
                        ),
                    }
                )

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    _append_record(item)
        elif isinstance(payload, dict):
            entities = payload.get("entities")
            relationships = payload.get("relationships")

            if isinstance(entities, list):
                for item in entities:
                    if isinstance(item, dict):
                        enriched = {"record_type": "entity", **item}
                        _append_record(enriched)

            if isinstance(relationships, list):
                for item in relationships:
                    if isinstance(item, dict):
                        enriched = {"record_type": "relationship", **item}
                        _append_record(enriched)

            if not records:
                _append_record(payload)

        if records:
            return records

    return None


def _parse_records_from_text_scan(
    text: str,
    tuple_delimiter: str,
) -> list[dict[str, Any]]:
    delimiter = re.escape(tuple_delimiter)

    entity_re = re.compile(
        rf'\(\s*["\']entity["\']\s*{delimiter}\s*(.*?)\s*{delimiter}\s*(.*?)\s*{delimiter}\s*(.*?)\)',
        re.IGNORECASE | re.DOTALL,
    )
    relationship_re = re.compile(
        rf'\(\s*["\']relationship["\']\s*{delimiter}\s*(.*?)\s*{delimiter}\s*(.*?)\s*{delimiter}\s*(.*?)\s*{delimiter}\s*(.*?)\)',
        re.IGNORECASE | re.DOTALL,
    )

    located: list[tuple[int, dict[str, Any]]] = []

    for match in entity_re.finditer(text):
        located.append(
            (
                match.start(),
                {
                    "record_type": "entity",
                    "entity_name": _normalize_whitespace(match.group(1)),
                    "entity_type": _normalize_whitespace(match.group(2)).upper(),
                    "entity_description": _normalize_whitespace(match.group(3)),
                },
            )
        )

    for match in relationship_re.finditer(text):
        located.append(
            (
                match.start(),
                {
                    "record_type": "relationship",
                    "source_entity": _normalize_whitespace(match.group(1)),
                    "target_entity": _normalize_whitespace(match.group(2)),
                    "relationship_description": _normalize_whitespace(match.group(3)),
                    "relationship_strength": _normalize_strength(match.group(4)),
                },
            )
        )

    located.sort(key=lambda item: item[0])
    return [record for _, record in located]


def parse_extraction_output(
    output: str,
    tuple_delimiter: str = DEFAULT_TUPLE_DELIMITER,
    record_delimiter: str = DEFAULT_RECORD_DELIMITER,
    completion_delimiter: str = DEFAULT_COMPLETION_DELIMITER,
) -> list[dict[str, Any]]:
    cleaned = _strip_code_fence(output)
    if not cleaned:
        return []

    json_records = _parse_json_records(cleaned)
    if json_records:
        return json_records

    if completion_delimiter in cleaned:
        cleaned = cleaned.split(completion_delimiter, 1)[0]

    candidates = [
        item.strip() for item in cleaned.split(record_delimiter) if item.strip()
    ]
    if not candidates:
        candidates = [line.strip() for line in cleaned.splitlines() if line.strip()]

    parsed_records: list[dict[str, Any]] = []
    for candidate in candidates:
        entity_record = _parse_entity_tuple(candidate, tuple_delimiter)
        if entity_record is not None:
            parsed_records.append(entity_record)
            continue

        relationship_record = _parse_relationship_tuple(candidate, tuple_delimiter)
        if relationship_record is not None:
            parsed_records.append(relationship_record)

    if parsed_records:
        return parsed_records

    return _parse_records_from_text_scan(cleaned, tuple_delimiter)


def _build_prompt_from_messages(messages: Sequence[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())

    prompt = "\n\n".join(parts).strip()
    if not prompt:
        raise ValueError("No text content found in messages")
    return prompt


def _get_default_mistral_service() -> MistralService:
    global _DEFAULT_MISTRAL_SERVICE
    if _DEFAULT_MISTRAL_SERVICE is None:
        _DEFAULT_MISTRAL_SERVICE = _create_mistral_service(
            model_dir=DEFAULT_MODEL_DIR,
            default_max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            default_temperature=DEFAULT_TEMPERATURE,
        )
    return _DEFAULT_MISTRAL_SERVICE


def chat(
    messages: Sequence[dict[str, Any]],
    model: str = "mistral",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    normalized_model = model.strip().lower()
    if "mistral" not in normalized_model:
        raise ValueError(
            f"Unsupported model '{model}'. This extractor uses the local Mistral model."
        )

    prompt = _build_prompt_from_messages(messages)
    service = _get_default_mistral_service()
    return service.generate_answer(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def _extract_with_service(
    service: MistralService,
    text: str,
    company_name: str | None,
    entity_types: Sequence[str],
    max_new_tokens: int | None,
    temperature: float | None,
    tuple_delimiter: str,
    record_delimiter: str,
    completion_delimiter: str,
) -> tuple[str, list[dict[str, Any]]]:
    prompt = create_extraction_prompt(
        entity_types=entity_types,
        text=text,
        company_name=company_name,
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        completion_delimiter=completion_delimiter,
    )

    output = service.generate_answer(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    records = parse_extraction_output(
        output,
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        completion_delimiter=completion_delimiter,
    )
    return output, records


def extract_entities(text: str) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "user",
            "content": create_extraction_prompt(ENTITY_TYPES, text),
        }
    ]
    output = chat(messages, model="mistral")
    return parse_extraction_output(output)


def extract_entities_for_company(text: str, company_name: str) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "user",
            "content": create_extraction_prompt(
                ENTITY_TYPES,
                text,
                company_name=company_name,
            ),
        }
    ]
    output = chat(messages, model="mistral")
    return parse_extraction_output(output)


def _normalize_record(
    record: dict[str, Any],
    company_name: str,
    allowed_entity_types: set[str],
) -> dict[str, Any] | None:
    record_type = str(record.get("record_type", "")).lower().strip()

    if record_type == "entity":
        entity_name = _canonical_entity_name(
            str(record.get("entity_name", "")), company_name
        )
        entity_type = _normalize_whitespace(str(record.get("entity_type", "")).upper())
        entity_description = _normalize_whitespace(
            str(record.get("entity_description", ""))
        )

        if (
            not entity_name
            or not entity_type
            or entity_type not in allowed_entity_types
            or _is_probably_noise_entity_name(entity_name, entity_type)
        ):
            return None

        return {
            "record_type": "entity",
            "entity_name": entity_name,
            "entity_type": entity_type,
            "entity_description": entity_description,
            "company": company_name,
        }

    if record_type == "relationship":
        source_entity = _canonical_entity_name(
            str(record.get("source_entity", "")),
            company_name,
        )
        target_entity = _canonical_entity_name(
            str(record.get("target_entity", "")),
            company_name,
        )
        relationship_description = _normalize_whitespace(
            str(record.get("relationship_description", ""))
        )

        if not source_entity or not target_entity:
            return None

        return {
            "record_type": "relationship",
            "source_entity": source_entity,
            "target_entity": target_entity,
            "relationship_description": relationship_description,
            "relationship_strength": _normalize_strength(
                record.get("relationship_strength", 5)
            ),
            "company": company_name,
        }

    return None


def _append_source(
    sources: list[dict[str, int | str]],
    source_file: str,
    page_number: int,
    unit_index: int,
) -> None:
    source_key = (source_file, page_number, unit_index)
    for source in sources:
        existing_key = (
            str(source.get("source_file", "")),
            int(source.get("page_number", -1)),
            int(source.get("unit_index", -1)),
        )
        if existing_key == source_key:
            return

    sources.append(
        {
            "source_file": source_file,
            "page_number": page_number,
            "unit_index": unit_index,
        }
    )


def _merge_entity(
    store: dict[tuple[str, str], dict[str, Any]],
    entity: dict[str, Any],
    source_file: str,
    page_number: int,
    unit_index: int,
) -> None:
    key = (str(entity["entity_name"]), str(entity["entity_type"]))
    existing = store.get(key)
    if existing is None:
        store[key] = {
            **entity,
            "sources": [
                {
                    "source_file": source_file,
                    "page_number": page_number,
                    "unit_index": unit_index,
                }
            ],
        }
        return

    if len(str(entity.get("entity_description", ""))) > len(
        str(existing.get("entity_description", ""))
    ):
        existing["entity_description"] = entity.get("entity_description", "")

    _append_source(existing["sources"], source_file, page_number, unit_index)


def _merge_relationship(
    store: dict[tuple[str, str, str], dict[str, Any]],
    relationship: dict[str, Any],
    source_file: str,
    page_number: int,
    unit_index: int,
) -> None:
    key = (
        str(relationship["source_entity"]),
        str(relationship["target_entity"]),
        str(relationship.get("relationship_description", "")),
    )
    existing = store.get(key)
    if existing is None:
        store[key] = {
            **relationship,
            "sources": [
                {
                    "source_file": source_file,
                    "page_number": page_number,
                    "unit_index": unit_index,
                }
            ],
        }
        return

    existing["relationship_strength"] = max(
        _normalize_strength(existing.get("relationship_strength", 1)),
        _normalize_strength(relationship.get("relationship_strength", 1)),
    )

    if len(str(relationship.get("relationship_description", ""))) > len(
        str(existing.get("relationship_description", ""))
    ):
        existing["relationship_description"] = relationship.get(
            "relationship_description", ""
        )

    _append_source(existing["sources"], source_file, page_number, unit_index)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=True))
            output_file.write("\n")


def _build_extraction_units(
    pages: Sequence[PageSection],
    max_chars_per_unit: int,
) -> list[ExtractionUnit]:
    units: list[ExtractionUnit] = []

    for page in pages:
        clean_text = _clean_page_text(page.text)
        if not clean_text:
            continue

        split_units = split_text_units(
            clean_text, max_chars_per_unit=max_chars_per_unit
        )
        for unit_index, unit_text in enumerate(split_units, start=1):
            units.append(
                ExtractionUnit(
                    page_number=page.page_number,
                    unit_index=unit_index,
                    text=unit_text,
                )
            )

    return units


def _row_unit_key(page_number: int, unit_index: int) -> tuple[int, int]:
    return int(page_number), int(unit_index)


def _load_existing_rows_by_unit(
    records_raw_path: Path,
) -> dict[tuple[int, int], dict[str, Any]]:
    if not records_raw_path.exists():
        return {}

    rows_by_unit: dict[tuple[int, int], dict[str, Any]] = {}

    for line in records_raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(row, dict):
            continue

        page_number = row.get("page_number")
        unit_index = row.get("unit_index")
        if page_number is None or unit_index is None:
            continue

        key = _row_unit_key(int(page_number), int(unit_index))
        rows_by_unit[key] = row

    return rows_by_unit


def _normalize_records_for_company(
    records: Sequence[dict[str, Any]],
    company_name: str,
    allowed_entity_types: set[str],
) -> tuple[list[dict[str, Any]], int]:
    normalized_records: list[dict[str, Any]] = []
    skipped_records = 0

    for record in records:
        if not isinstance(record, dict):
            skipped_records += 1
            continue

        normalized = _normalize_record(
            record=record,
            company_name=company_name,
            allowed_entity_types=allowed_entity_types,
        )
        if normalized is None:
            skipped_records += 1
            continue

        normalized_records.append(normalized)

    return normalized_records, skipped_records


def _normalize_records_for_row(
    row: dict[str, Any],
    company_name: str,
    allowed_entity_types: set[str],
) -> tuple[list[dict[str, Any]], int]:
    candidate_records: list[dict[str, Any]] = []

    parsed_records = row.get("parsed_records")
    if isinstance(parsed_records, list) and parsed_records:
        candidate_records = [
            record for record in parsed_records if isinstance(record, dict)
        ]
    else:
        normalized_records = row.get("normalized_records")
        if isinstance(normalized_records, list):
            candidate_records = [
                record for record in normalized_records if isinstance(record, dict)
            ]

    return _normalize_records_for_company(
        records=candidate_records,
        company_name=company_name,
        allowed_entity_types=allowed_entity_types,
    )


def _merge_normalized_records(
    normalized_records: Sequence[dict[str, Any]],
    entity_store: dict[tuple[str, str], dict[str, Any]],
    relationship_store: dict[tuple[str, str, str], dict[str, Any]],
    source_file: str,
    page_number: int,
    unit_index: int,
) -> None:
    for record in normalized_records:
        if record["record_type"] == "entity":
            _merge_entity(
                store=entity_store,
                entity=record,
                source_file=source_file,
                page_number=page_number,
                unit_index=unit_index,
            )
        elif record["record_type"] == "relationship":
            _merge_relationship(
                store=relationship_store,
                relationship=record,
                source_file=source_file,
                page_number=page_number,
                unit_index=unit_index,
            )


def extract_from_markdown_file(
    markdown_file: Path,
    output_dir: Path,
    mistral_service: MistralService,
    entity_types: Sequence[str] = ENTITY_TYPES,
    max_chars_per_unit: int = DEFAULT_MAX_CHARS_PER_UNIT,
    max_new_tokens: int | None = DEFAULT_MAX_NEW_TOKENS,
    temperature: float | None = DEFAULT_TEMPERATURE,
    tuple_delimiter: str = DEFAULT_TUPLE_DELIMITER,
    record_delimiter: str = DEFAULT_RECORD_DELIMITER,
    completion_delimiter: str = DEFAULT_COMPLETION_DELIMITER,
    max_pages: int | None = None,
    resume_from_existing: bool = False,
    retry_failed: bool = True,
    stop_on_fatal_model_error: bool = True,
    checkpoint_every_n_units: int = 10,
    verbose: bool = False,
) -> dict[str, Any]:
    company_slug = markdown_file.stem.lower().strip()
    company_name = company_slug.upper()
    company_output_dir = output_dir / company_slug
    records_raw_path = company_output_dir / "records_raw.jsonl"

    markdown_text = markdown_file.read_text(encoding="utf-8")
    pages = parse_markdown_pages(markdown_text)
    if max_pages is not None and max_pages > 0:
        pages = pages[:max_pages]

    units = _build_extraction_units(
        pages=pages,
        max_chars_per_unit=max_chars_per_unit,
    )

    existing_rows_by_unit: dict[tuple[int, int], dict[str, Any]] = {}
    if resume_from_existing:
        existing_rows_by_unit = _load_existing_rows_by_unit(records_raw_path)
        if verbose:
            print(
                f"[{company_slug}] loaded {len(existing_rows_by_unit)} existing unit records for resume"
            )

    entity_store: dict[tuple[str, str], dict[str, Any]] = {}
    relationship_store: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []

    allowed_entity_types = {
        _normalize_whitespace(value).upper() for value in entity_types
    }

    attempted_units = 0
    reused_completed_units = 0
    reused_failed_units = 0
    skipped_records = 0
    parse_failures = 0
    model_failures = 0
    aborted = False
    fatal_error_message: str | None = None

    started = perf_counter()

    for unit in units:
        unit_key = _row_unit_key(unit.page_number, unit.unit_index)
        existing_row = existing_rows_by_unit.get(unit_key)

        if existing_row is not None:
            existing_input_text = _normalize_whitespace(
                str(existing_row.get("input_text", ""))
            )
            current_input_text = _normalize_whitespace(unit.text)

            if existing_input_text == current_input_text:
                existing_error = existing_row.get("error")

                if existing_error is None:
                    normalized_records, skipped_from_row = _normalize_records_for_row(
                        row=existing_row,
                        company_name=company_name,
                        allowed_entity_types=allowed_entity_types,
                    )
                    skipped_records += skipped_from_row

                    row_copy = {
                        "company": company_name,
                        "source_file": markdown_file.name,
                        "page_number": unit.page_number,
                        "unit_index": unit.unit_index,
                        "input_text": unit.text,
                        "raw_output": str(existing_row.get("raw_output", "")),
                        "parsed_records": existing_row.get("parsed_records", []),
                        "normalized_records": normalized_records,
                        "error": None,
                    }

                    raw_rows.append(row_copy)
                    reused_completed_units += 1

                    _merge_normalized_records(
                        normalized_records=normalized_records,
                        entity_store=entity_store,
                        relationship_store=relationship_store,
                        source_file=markdown_file.name,
                        page_number=unit.page_number,
                        unit_index=unit.unit_index,
                    )

                    if verbose:
                        print(
                            f"[{company_slug}] page {unit.page_number} unit {unit.unit_index}: reused completed result"
                        )

                    if (
                        checkpoint_every_n_units > 0
                        and len(raw_rows) % checkpoint_every_n_units == 0
                    ):
                        _write_jsonl(records_raw_path, raw_rows)

                    continue

                if not retry_failed:
                    row_copy = {
                        "company": company_name,
                        "source_file": markdown_file.name,
                        "page_number": unit.page_number,
                        "unit_index": unit.unit_index,
                        "input_text": unit.text,
                        "raw_output": str(existing_row.get("raw_output", "")),
                        "parsed_records": existing_row.get("parsed_records", []),
                        "normalized_records": existing_row.get(
                            "normalized_records", []
                        ),
                        "error": str(existing_error),
                    }
                    raw_rows.append(row_copy)
                    reused_failed_units += 1

                    if verbose:
                        print(
                            f"[{company_slug}] page {unit.page_number} unit {unit.unit_index}: kept previous failure"
                        )

                    if (
                        checkpoint_every_n_units > 0
                        and len(raw_rows) % checkpoint_every_n_units == 0
                    ):
                        _write_jsonl(records_raw_path, raw_rows)

                    continue

        attempted_units += 1

        try:
            raw_output, parsed_records = _extract_with_service(
                service=mistral_service,
                text=unit.text,
                company_name=company_name,
                entity_types=entity_types,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                tuple_delimiter=tuple_delimiter,
                record_delimiter=record_delimiter,
                completion_delimiter=completion_delimiter,
            )
        except Exception as exc:
            error_message = str(exc)
            model_failures += 1

            raw_rows.append(
                {
                    "company": company_name,
                    "source_file": markdown_file.name,
                    "page_number": unit.page_number,
                    "unit_index": unit.unit_index,
                    "input_text": unit.text,
                    "raw_output": "",
                    "parsed_records": [],
                    "normalized_records": [],
                    "error": error_message,
                }
            )

            if verbose:
                print(
                    f"[{company_slug}] page {unit.page_number} unit {unit.unit_index}: model call failed"
                )

            if (
                checkpoint_every_n_units > 0
                and len(raw_rows) % checkpoint_every_n_units == 0
            ):
                _write_jsonl(records_raw_path, raw_rows)

            if stop_on_fatal_model_error and _is_fatal_model_error(error_message):
                aborted = True
                fatal_error_message = error_message
                if verbose:
                    print(
                        f"[{company_slug}] fatal model error detected, stopping early at page {unit.page_number} unit {unit.unit_index}"
                    )
                break

            continue

        normalized_records, skipped_from_row = _normalize_records_for_company(
            records=parsed_records,
            company_name=company_name,
            allowed_entity_types=allowed_entity_types,
        )
        skipped_records += skipped_from_row

        _merge_normalized_records(
            normalized_records=normalized_records,
            entity_store=entity_store,
            relationship_store=relationship_store,
            source_file=markdown_file.name,
            page_number=unit.page_number,
            unit_index=unit.unit_index,
        )

        if raw_output.strip() and not parsed_records:
            parse_failures += 1

        raw_rows.append(
            {
                "company": company_name,
                "source_file": markdown_file.name,
                "page_number": unit.page_number,
                "unit_index": unit.unit_index,
                "input_text": unit.text,
                "raw_output": raw_output,
                "parsed_records": parsed_records,
                "normalized_records": normalized_records,
                "error": None,
            }
        )

        if verbose:
            print(
                f"[{company_slug}] page {unit.page_number} unit {unit.unit_index}: "
                f"parsed={len(parsed_records)} normalized={len(normalized_records)}"
            )

        if (
            checkpoint_every_n_units > 0
            and len(raw_rows) % checkpoint_every_n_units == 0
        ):
            _write_jsonl(records_raw_path, raw_rows)

    entities = sorted(
        entity_store.values(),
        key=lambda item: (
            str(item.get("entity_type", "")),
            str(item.get("entity_name", "")),
        ),
    )
    relationships = list(relationship_store.values())

    entity_names = {str(item.get("entity_name", "")) for item in entities}
    relationships_before_orphan_filter = len(relationships)
    relationships = [
        relationship
        for relationship in relationships
        if str(relationship.get("source_entity", "")) in entity_names
        and str(relationship.get("target_entity", "")) in entity_names
    ]
    orphan_relationships_filtered = relationships_before_orphan_filter - len(
        relationships
    )

    relationships = sorted(
        relationships,
        key=lambda item: (
            str(item.get("source_entity", "")),
            str(item.get("target_entity", "")),
            -int(item.get("relationship_strength", 0)),
        ),
    )

    for item in entities + relationships:
        sources = item.get("sources", [])
        if isinstance(sources, list):
            sources.sort(
                key=lambda source: (
                    int(source.get("page_number", -1)),
                    int(source.get("unit_index", -1)),
                )
            )

    elapsed_seconds = round(perf_counter() - started, 2)

    _write_json(company_output_dir / "entities.json", entities)
    _write_json(company_output_dir / "relationships.json", relationships)
    _write_json(
        company_output_dir / "graph_records.json",
        {
            "company": company_name,
            "source_file": markdown_file.name,
            "entities": entities,
            "relationships": relationships,
        },
    )
    _write_jsonl(company_output_dir / "records_raw.jsonl", raw_rows)

    model_failures = sum(1 for row in raw_rows if row.get("error"))
    parse_failures = sum(
        1
        for row in raw_rows
        if row.get("error") is None
        and str(row.get("raw_output", "")).strip()
        and not row.get("parsed_records")
    )

    summary = {
        "company": company_name,
        "source_file": markdown_file.name,
        "page_count": len(pages),
        "units_planned": len(units),
        "units_processed": len(raw_rows),
        "units_attempted": attempted_units,
        "units_reused_completed": reused_completed_units,
        "units_reused_failed": reused_failed_units,
        "entities_count": len(entities),
        "relationships_count": len(relationships),
        "model_failures": model_failures,
        "parse_failures": parse_failures,
        "skipped_records": skipped_records,
        "orphan_relationships_filtered": orphan_relationships_filtered,
        "resume_mode": resume_from_existing,
        "retry_failed": retry_failed,
        "aborted": aborted,
        "fatal_error_message": fatal_error_message,
        "elapsed_seconds": elapsed_seconds,
        "output_dir": str(company_output_dir),
    }
    _write_json(company_output_dir / "run_summary.json", summary)

    return summary


def run_batch_extraction(
    markdown_dir: Path,
    output_dir: Path,
    companies: Sequence[str] | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    max_chars_per_unit: int = DEFAULT_MAX_CHARS_PER_UNIT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_pages: int | None = None,
    resume_from_existing: bool = False,
    retry_failed: bool = True,
    stop_on_fatal_model_error: bool = True,
    checkpoint_every_n_units: int = 10,
    verbose: bool = False,
) -> dict[str, Any]:
    if not markdown_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {markdown_dir}")

    markdown_files = sorted(markdown_dir.glob("*.md"))
    if companies:
        requested = {
            company.strip().lower() for company in companies if company.strip()
        }
        markdown_files = [
            markdown_file
            for markdown_file in markdown_files
            if markdown_file.stem.lower() in requested
        ]

    if not markdown_files:
        raise RuntimeError(f"No markdown files selected in {markdown_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    mistral_service = _create_mistral_service(
        model_dir=model_dir,
        default_max_new_tokens=max_new_tokens,
        default_temperature=temperature,
    )

    summaries: list[dict[str, Any]] = []
    total_entities = 0
    total_relationships = 0
    total_units = 0
    aborted = False
    abort_reason: str | None = None

    run_started = perf_counter()

    for markdown_file in markdown_files:
        print(f"Processing {markdown_file.name} ...")
        summary = extract_from_markdown_file(
            markdown_file=markdown_file,
            output_dir=output_dir,
            mistral_service=mistral_service,
            entity_types=ENTITY_TYPES,
            max_chars_per_unit=max_chars_per_unit,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            max_pages=max_pages,
            resume_from_existing=resume_from_existing,
            retry_failed=retry_failed,
            stop_on_fatal_model_error=stop_on_fatal_model_error,
            checkpoint_every_n_units=checkpoint_every_n_units,
            verbose=verbose,
        )
        summaries.append(summary)

        total_entities += int(summary.get("entities_count", 0))
        total_relationships += int(summary.get("relationships_count", 0))
        total_units += int(summary.get("units_processed", 0))

        print(
            f"Done {markdown_file.name}: entities={summary['entities_count']} "
            f"relationships={summary['relationships_count']}"
        )

        if summary.get("aborted") and stop_on_fatal_model_error:
            aborted = True
            abort_reason = (
                f"{markdown_file.name}: "
                f"{str(summary.get('fatal_error_message') or 'fatal model error')}"
            )
            print(
                f"Stopping batch extraction after fatal model error in {markdown_file.name}."
            )
            break

    run_elapsed = round(perf_counter() - run_started, 2)
    run_summary = {
        "aborted": aborted,
        "abort_reason": abort_reason,
        "companies_processed": len(summaries),
        "companies_requested": len(markdown_files),
        "total_units_processed": total_units,
        "total_entities": total_entities,
        "total_relationships": total_relationships,
        "elapsed_seconds": run_elapsed,
        "companies": summaries,
    }

    _write_json(output_dir / "run_summary.json", run_summary)
    return run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract GraphRAG-style entities and relationships from markdown ESG reports "
            "with local Mistral and save company-specific outputs."
        )
    )
    parser.add_argument(
        "--markdown-dir",
        default=str(DATA_MARKDOWN_DIR),
        help="Directory containing markdown reports (*.md).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where extraction artifacts will be stored.",
    )
    parser.add_argument(
        "--companies",
        nargs="+",
        default=None,
        help="Optional list of company stems to process (e.g. nvidia google).",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Local Mistral model directory.",
    )
    parser.add_argument(
        "--max-chars-per-unit",
        type=int,
        default=DEFAULT_MAX_CHARS_PER_UNIT,
        help="Maximum characters per extraction text unit.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generation tokens per extraction call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Generation temperature for local Mistral.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page cap per company file (useful for quick tests).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from existing records_raw.jsonl files by skipping completed "
            "units for each company."
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --resume is enabled, retry previously failed units. "
            "Use --no-retry-failed to keep prior failures unchanged."
        ),
    )
    parser.add_argument(
        "--stop-on-fatal-model-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stop processing remaining units and companies when a fatal CUDA/model "
            "error is detected."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-n-units",
        type=int,
        default=10,
        help=(
            "Persist records_raw.jsonl every N units for resumability. Set 0 to "
            "disable intermediate checkpoints."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print page/unit-level extraction progress.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_chars_per_unit < 500:
        parser.error("--max-chars-per-unit must be >= 500")
    if args.max_new_tokens < 64:
        parser.error("--max-new-tokens must be >= 64")
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages must be >= 1")
    if args.checkpoint_every_n_units < 0:
        parser.error("--checkpoint-every-n-units must be >= 0")

    summary = run_batch_extraction(
        markdown_dir=Path(args.markdown_dir),
        output_dir=Path(args.output_dir),
        companies=args.companies,
        model_dir=Path(args.model_dir),
        max_chars_per_unit=args.max_chars_per_unit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_pages=args.max_pages,
        resume_from_existing=args.resume,
        retry_failed=args.retry_failed,
        stop_on_fatal_model_error=args.stop_on_fatal_model_error,
        checkpoint_every_n_units=args.checkpoint_every_n_units,
        verbose=args.verbose,
    )

    print(
        "Extraction complete: "
        f"companies={summary['companies_processed']} "
        f"entities={summary['total_entities']} "
        f"relationships={summary['total_relationships']}"
    )

    if summary.get("aborted"):
        print(f"Extraction stopped early: {summary.get('abort_reason')}")


if __name__ == "__main__":
    main()
