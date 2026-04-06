from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    companies: list[str] = Field(default_factory=list)
    top_k_per_company: int = Field(default=5, ge=1, le=20)
    max_new_tokens: int = Field(default=512, ge=64, le=2048)
    temperature: float = Field(default=0.1, ge=0.0, le=1.5)


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


app = FastAPI(title="Graph RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str | bool]:
    rag_service: RagService = app.state.rag_service
    mistral_service: MistralService = app.state.mistral_service

    try:
        rag_service.verify_connectivity()
        neo4j_ok = True
    except Exception:
        neo4j_ok = False

    return {
        "status": "ok" if neo4j_ok else "degraded",
        "neo4j_connected": neo4j_ok,
        "mistral_loaded": mistral_service.model_loaded,
    }


@app.get("/companies")
def companies() -> dict[str, list[str]]:
    rag_service: RagService = app.state.rag_service
    try:
        return {"companies": rag_service.list_companies()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    rag_service: RagService = app.state.rag_service
    mistral_service: MistralService = app.state.mistral_service

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        available_companies = rag_service.list_companies()
        requested_companies = request.companies or available_companies

        chunks_by_company = rag_service.retrieve_by_company(
            question=question,
            companies=requested_companies,
            top_k_per_company=request.top_k_per_company,
        )

        total_chunks = sum(len(chunks) for chunks in chunks_by_company.values())
        if total_chunks == 0:
            resolved_companies = list(chunks_by_company.keys()) or requested_companies
            answer = "I cannot find this in the provided report excerpts."
            response_chunks: dict[str, list[ChunkResponse]] = {
                company: [] for company in resolved_companies
            }
            return ChatResponse(
                answer=answer,
                companies=resolved_companies,
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
            company: [ChunkResponse(**chunk.to_dict()) for chunk in chunks]
            for company, chunks in chunks_by_company.items()
        }

        return ChatResponse(
            answer=answer,
            companies=list(chunks_by_company.keys()),
            chunks=response_chunks,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
