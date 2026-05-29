from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.graphrag_service import (
    CompanyGraphEvidence,
    GraphRagService,
)
from app.mistral_service import MistralService
from app.rag_service import RagService, RetrievedChunk
from paths import MODELS_DIR


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


def _truncate_text(text: str, max_chars: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    truncated = clean[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return f"{truncated} ..."


def _build_rag_prompt(
    question: str,
    chunks_by_company: dict[str, list[RetrievedChunk]],
    max_chars_per_chunk: int,
) -> str:
    context_blocks: list[str] = []
    for company, chunks in chunks_by_company.items():
        for chunk in chunks:
            header = (
                f"[company={company}; source={chunk.source_file}; "
                f"page={chunk.page_number_raw}; score={chunk.score:.4f}]"
            )
            body = _truncate_text(chunk.text, max_chars=max_chars_per_chunk)
            context_blocks.append(f"{header}\n{body}")

    context_text = "\n\n".join(context_blocks)
    if not context_text:
        context_text = "No relevant report excerpts were found."

    return (
        "You are an ESG reporting assistant. "
        "Answer only using the provided context from company reports.\n"
        "If the answer is missing from context, say exactly: "
        "I cannot find this in the provided report excerpts.\n"
        "For factual claims, add citations in the form [company page_number].\n"
        "For compare questions, structure the answer by company with a short conclusion.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer:"
    )


def _normalize_requested_companies(
    available_companies: list[str], requested_companies: list[str]
) -> list[str]:
    available = [company.strip().lower() for company in available_companies if company]
    available = list(dict.fromkeys(available))
    requested = [company.strip().lower() for company in requested_companies if company]

    if not requested:
        return available

    allowed = set(available)
    filtered = [company for company in requested if company in allowed]
    filtered = list(dict.fromkeys(filtered))
    return filtered or available


def _build_graphrag_prompt(
    question: str,
    evidence_by_company: dict[str, CompanyGraphEvidence],
    max_chars_per_chunk: int,
) -> str:
    context_blocks: list[str] = []

    for company, evidence in evidence_by_company.items():
        entity_lines: list[str] = []
        for entity in evidence.entities:
            description = _truncate_text(entity.entity_description, max_chars=280)
            refs = ", ".join(entity.source_refs[:3]) if entity.source_refs else "none"
            entity_lines.append(
                "- "
                f"{entity.entity_name} "
                f"(type={entity.entity_type}, degree={entity.degree}, "
                f"freq={entity.frequency}, score={entity.score:.4f}, refs={refs}) "
                f"{description}"
            )

        relationship_lines: list[str] = []
        for relationship in evidence.relationships:
            description = _truncate_text(
                relationship.relationship_description,
                max_chars=280,
            )
            refs = (
                ", ".join(relationship.source_refs[:3])
                if relationship.source_refs
                else "none"
            )
            relationship_lines.append(
                "- "
                f"{relationship.source_entity} -> {relationship.target_entity} "
                f"(weight={relationship.relationship_weight:.2f}, "
                f"combined_degree={relationship.combined_degree}, "
                f"score={relationship.score:.4f}, refs={refs}) "
                f"{description}"
            )

        chunk_lines: list[str] = []
        for chunk in evidence.chunks:
            header = (
                f"[company={company}; source={chunk.source_file}; "
                f"page={chunk.page_number_raw}; score={chunk.score:.4f}]"
            )
            body = _truncate_text(chunk.text, max_chars=max_chars_per_chunk)
            chunk_lines.append(f"{header}\n{body}")

        entity_section = "\n".join(entity_lines) if entity_lines else "- none"
        relationship_section = (
            "\n".join(relationship_lines) if relationship_lines else "- none"
        )
        chunk_section = "\n\n".join(chunk_lines) if chunk_lines else "- none"

        context_blocks.append(
            f"[company={company}]\n"
            f"Entities:\n{entity_section}\n\n"
            f"Relationships:\n{relationship_section}\n\n"
            f"Supporting Excerpts:\n{chunk_section}"
        )

    context_text = "\n\n".join(context_blocks)
    if not context_text:
        context_text = "No graph evidence was found for the requested companies."

    return (
        "You are an ESG reporting assistant using graph-structured evidence. "
        "Answer only from the provided entities, relationships, and report excerpts.\n"
        "If the answer is missing from the evidence, say exactly: "
        "I cannot find this in the provided report excerpts.\n"
        "For factual claims, add citations in the form [company page_number].\n"
        "When comparing companies, reason step by step by company and end with a concise conclusion.\n\n"
        f"Question:\n{question}\n\n"
        f"Graph Context:\n{context_text}\n\n"
        "Answer:"
    )


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    companies: list[str] = Field(default_factory=list)
    top_k_per_company: int = Field(default=5, ge=1, le=20)
    max_new_tokens: int = Field(default=512, ge=64, le=2048)
    temperature: float = Field(default=0.1, ge=0.0, le=1.5)


class GraphChatRequest(ChatRequest):
    top_entities_per_company: int = Field(default=8, ge=1, le=50)
    top_relationships_per_company: int = Field(default=20, ge=1, le=80)
    top_graph_chunks_per_company: int = Field(default=5, ge=1, le=20)


class ChunkResponse(BaseModel):
    id: str
    company: str
    source_file: str
    page_number: int
    page_number_raw: str
    page_ordinal: int
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    companies: list[str]
    chunks: dict[str, list[ChunkResponse]]


class GraphEntityResponse(BaseModel):
    company: str
    entity_name: str
    entity_type: str
    entity_description: str
    degree: int
    frequency: int
    source_refs: list[str]
    score: float


class GraphRelationshipResponse(BaseModel):
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


class GraphChatResponse(BaseModel):
    answer: str
    companies: list[str]
    entities: dict[str, list[GraphEntityResponse]]
    relationships: dict[str, list[GraphRelationshipResponse]]
    chunks: dict[str, list[ChunkResponse]]


class CompareResponse(BaseModel):
    question: str
    companies: list[str]
    rag: ChatResponse
    graphrag: GraphChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    neo4j_user, neo4j_password = _resolve_neo4j_auth()

    app.state.rag_service = RagService(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2"
        ),
    )
    app.state.graphrag_service = GraphRagService(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2"
        ),
    )
    app.state.mistral_service = MistralService(
        model_dir=Path(
            os.getenv("MISTRAL_MODEL_DIR", str(MODELS_DIR / "Mistral-7B-Instruct-v0.3"))
        ),
        default_max_new_tokens=int(os.getenv("LLM_DEFAULT_MAX_NEW_TOKENS", "512")),
        default_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1")),
    )

    try:
        yield
    finally:
        app.state.rag_service.close()
        app.state.graphrag_service.close()


app = FastAPI(title="Graph RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str | bool]:
    rag_service: RagService = app.state.rag_service
    graphrag_service: GraphRagService = app.state.graphrag_service
    mistral_service: MistralService = app.state.mistral_service

    try:
        rag_service.verify_connectivity()
        graphrag_service.verify_connectivity()
        neo4j_ok = True
    except Exception:
        neo4j_ok = False

    return {
        "status": "ok" if neo4j_ok else "degraded",
        "neo4j_connected": neo4j_ok,
        "graphrag_ready": neo4j_ok,
        "mistral_loaded": mistral_service.model_loaded,
    }


@app.get("/companies")
def companies() -> dict[str, list[str]]:
    rag_service: RagService = app.state.rag_service
    try:
        return {"companies": rag_service.list_companies()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _chunk_to_response(chunk: RetrievedChunk) -> ChunkResponse:
    return ChunkResponse(
        id=str(chunk.id),
        company=str(chunk.company),
        source_file=str(chunk.source_file),
        page_number=int(chunk.page_number),
        page_number_raw=str(chunk.page_number_raw),
        page_ordinal=int(chunk.page_ordinal),
        score=float(chunk.score),
        text=str(chunk.text),
    )


def _run_rag_chat(
    request: ChatRequest,
    rag_service: RagService,
    mistral_service: MistralService,
) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    available_companies = rag_service.list_companies()
    requested_companies = _normalize_requested_companies(
        available_companies=available_companies,
        requested_companies=request.companies,
    )

    chunks_by_company = rag_service.retrieve_by_company(
        question=question,
        companies=requested_companies,
        top_k_per_company=request.top_k_per_company,
    )

    total_chunks = sum(len(chunks) for chunks in chunks_by_company.values())
    if total_chunks == 0:
        answer = "I cannot find this in the provided report excerpts."
        response_chunks: dict[str, list[ChunkResponse]] = {
            company: [] for company in requested_companies
        }
        return ChatResponse(
            answer=answer,
            companies=requested_companies,
            chunks=response_chunks,
        )

    prompt = _build_rag_prompt(
        question=question,
        chunks_by_company=chunks_by_company,
        max_chars_per_chunk=int(os.getenv("RAG_MAX_CHARS_PER_CHUNK", "2200")),
    )
    answer = mistral_service.generate_answer(
        prompt=prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )

    response_chunks = {
        company: [_chunk_to_response(chunk) for chunk in chunks]
        for company, chunks in chunks_by_company.items()
    }

    return ChatResponse(
        answer=answer,
        companies=list(chunks_by_company.keys()),
        chunks=response_chunks,
    )


def _run_graphrag_chat(
    request: GraphChatRequest,
    rag_service: RagService,
    graphrag_service: GraphRagService,
    mistral_service: MistralService,
) -> GraphChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    available_companies = rag_service.list_companies()
    requested_companies = _normalize_requested_companies(
        available_companies=available_companies,
        requested_companies=request.companies,
    )

    evidence_by_company = graphrag_service.retrieve_by_company(
        question=question,
        companies=requested_companies,
        top_entities_per_company=request.top_entities_per_company,
        top_relationships_per_company=request.top_relationships_per_company,
        top_chunks_per_company=request.top_graph_chunks_per_company,
    )

    total_graph_items = sum(
        len(evidence.entities) + len(evidence.relationships) + len(evidence.chunks)
        for evidence in evidence_by_company.values()
    )

    response_entities: dict[str, list[GraphEntityResponse]] = {}
    response_relationships: dict[str, list[GraphRelationshipResponse]] = {}
    response_chunks: dict[str, list[ChunkResponse]] = {}

    for company, evidence in evidence_by_company.items():
        response_entities[company] = [
            GraphEntityResponse(**entity.to_dict()) for entity in evidence.entities
        ]
        response_relationships[company] = [
            GraphRelationshipResponse(**relationship.to_dict())
            for relationship in evidence.relationships
        ]
        response_chunks[company] = [
            _chunk_to_response(chunk) for chunk in evidence.chunks
        ]

    if total_graph_items == 0:
        return GraphChatResponse(
            answer="I cannot find this in the provided report excerpts.",
            companies=requested_companies,
            entities={company: [] for company in requested_companies},
            relationships={company: [] for company in requested_companies},
            chunks={company: [] for company in requested_companies},
        )

    prompt = _build_graphrag_prompt(
        question=question,
        evidence_by_company=evidence_by_company,
        max_chars_per_chunk=int(os.getenv("RAG_MAX_CHARS_PER_CHUNK", "2200")),
    )
    answer = mistral_service.generate_answer(
        prompt=prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )

    return GraphChatResponse(
        answer=answer,
        companies=list(evidence_by_company.keys()),
        entities=response_entities,
        relationships=response_relationships,
        chunks=response_chunks,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    rag_service: RagService = app.state.rag_service
    mistral_service: MistralService = app.state.mistral_service

    try:
        return _run_rag_chat(
            request=request,
            rag_service=rag_service,
            mistral_service=mistral_service,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/graphrag", response_model=GraphChatResponse)
def chat_graphrag(request: GraphChatRequest) -> GraphChatResponse:
    rag_service: RagService = app.state.rag_service
    graphrag_service: GraphRagService = app.state.graphrag_service
    mistral_service: MistralService = app.state.mistral_service

    try:
        return _run_graphrag_chat(
            request=request,
            rag_service=rag_service,
            graphrag_service=graphrag_service,
            mistral_service=mistral_service,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/compare", response_model=CompareResponse)
def chat_compare(request: GraphChatRequest) -> CompareResponse:
    rag_service: RagService = app.state.rag_service
    graphrag_service: GraphRagService = app.state.graphrag_service
    mistral_service: MistralService = app.state.mistral_service

    try:
        rag_result = _run_rag_chat(
            request=request,
            rag_service=rag_service,
            mistral_service=mistral_service,
        )
        graphrag_result = _run_graphrag_chat(
            request=request,
            rag_service=rag_service,
            graphrag_service=graphrag_service,
            mistral_service=mistral_service,
        )

        combined_companies = list(
            dict.fromkeys(graphrag_result.companies + rag_result.companies)
        )
        return CompareResponse(
            question=request.question.strip(),
            companies=combined_companies,
            rag=rag_result,
            graphrag=graphrag_result,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
