from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import streamlit as st


API_BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")


def _api_get(path: str, timeout: int = 20) -> dict:
    request = urllib.request.Request(f"{API_BASE_URL}{path}", method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _api_post(path: str, data: dict, timeout: int = 600) -> dict:
    raw = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        f"{API_BASE_URL}{path}",
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _load_companies() -> list[str]:
    try:
        response = _api_get("/companies")
        return response.get("companies", [])
    except urllib.error.HTTPError as exc:
        st.error(f"HTTP error while loading companies: {exc}")
    except urllib.error.URLError as exc:
        st.error(f"Could not reach API at {API_BASE_URL}: {exc.reason}")
    except Exception as exc:
        st.error(f"Unexpected error while loading companies: {exc}")
    return []


def _render_chunks(chunks_by_company: dict, heading: str = "Retrieved Chunks") -> None:
    st.subheader(heading)
    for company, chunks in chunks_by_company.items():
        st.markdown(f"### {company}")
        if not chunks:
            st.info("No chunks retrieved for this company.")
            continue

        for chunk in chunks:
            score = float(chunk.get("score") or 0.0)
            title = (
                f"{chunk.get('source_file', 'unknown')} | "
                f"page {chunk.get('page_number_raw', '?')} | "
                f"score {score:.4f}"
            )
            with st.expander(title):
                st.write(chunk.get("text", ""))


def _render_graph_entities(entities_by_company: dict, max_rows: int = 12) -> None:
    st.subheader("Graph Entities")
    for company, entities in entities_by_company.items():
        st.markdown(f"### {company}")
        if not entities:
            st.info("No entity evidence retrieved for this company.")
            continue

        for entity in entities[:max_rows]:
            st.markdown(
                "- "
                f"`{entity.get('entity_name', '')}` "
                f"({entity.get('entity_type', '')}) "
                f"degree={entity.get('degree', 0)} "
                f"score={float(entity.get('score') or 0.0):.4f}"
            )
            description = str(entity.get("entity_description", "")).strip()
            if description:
                st.caption(description)


def _render_graph_relationships(
    relationships_by_company: dict,
    max_rows: int = 12,
) -> None:
    st.subheader("Graph Relationships")
    for company, relationships in relationships_by_company.items():
        st.markdown(f"### {company}")
        if not relationships:
            st.info("No relationship evidence retrieved for this company.")
            continue

        for relationship in relationships[:max_rows]:
            st.markdown(
                "- "
                f"`{relationship.get('source_entity', '')}` -> "
                f"`{relationship.get('target_entity', '')}` "
                f"weight={float(relationship.get('relationship_weight') or 0.0):.2f} "
                f"score={float(relationship.get('score') or 0.0):.4f}"
            )
            description = str(relationship.get("relationship_description", "")).strip()
            if description:
                st.caption(description)


def _render_rag_result(result: dict) -> None:
    st.subheader("Answer")
    st.write(result.get("answer", ""))
    _render_chunks(result.get("chunks", {}), heading="Retrieved Chunks")


def _render_graphrag_result(result: dict) -> None:
    st.subheader("Answer")
    st.write(result.get("answer", ""))
    _render_graph_entities(result.get("entities", {}))
    _render_graph_relationships(result.get("relationships", {}))
    _render_chunks(result.get("chunks", {}), heading="Supporting Chunks")


st.set_page_config(page_title="Graph RAG ESG QA", layout="wide")
st.title("Graph RAG ESG QA")
st.caption("FastAPI + Neo4j + Mistral-7B-Instruct-v0.3 | RAG vs GraphRAG compare")

with st.sidebar:
    st.subheader("Connection")
    st.code(API_BASE_URL)
    refresh_clicked = st.button("Refresh Companies")

if "companies" not in st.session_state or refresh_clicked:
    st.session_state.companies = _load_companies()

companies = st.session_state.companies
if not companies:
    st.warning(
        "No companies loaded from API. Make sure Neo4j is running and index data is populated."
    )

question = st.text_area(
    "Question",
    value="Compare scope 1 emissions of Google and NVIDIA.",
    height=120,
)

mode = st.radio(
    "Answer Mode",
    options=["Compare", "RAG", "GraphRAG"],
    horizontal=True,
)

selected_companies = st.multiselect(
    "Companies",
    options=companies,
    default=companies,
)

col_left, col_mid, col_right = st.columns(3)
with col_left:
    top_k_per_company = st.slider(
        "Top-k per company", min_value=1, max_value=10, value=5
    )
with col_mid:
    top_entities_per_company = st.slider(
        "Graph entities/company",
        min_value=1,
        max_value=30,
        value=8,
    )
with col_right:
    top_relationships_per_company = st.slider(
        "Graph relationships/company",
        min_value=1,
        max_value=50,
        value=20,
    )

col_graph, col_llm = st.columns(2)
with col_graph:
    top_graph_chunks_per_company = st.slider(
        "Graph chunks/company",
        min_value=1,
        max_value=10,
        value=5,
    )
with col_llm:
    max_new_tokens = st.slider(
        "Max new tokens", min_value=128, max_value=1024, value=512
    )

temperature = st.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05
)

if st.button("Ask"):
    if not question.strip():
        st.error("Question cannot be empty.")
    else:
        payload = {
            "question": question,
            "companies": selected_companies,
            "top_k_per_company": top_k_per_company,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

        if mode in {"GraphRAG", "Compare"}:
            payload["top_entities_per_company"] = top_entities_per_company
            payload["top_relationships_per_company"] = top_relationships_per_company
            payload["top_graph_chunks_per_company"] = top_graph_chunks_per_company

        try:
            endpoint = "/chat/compare"
            if mode == "RAG":
                endpoint = "/chat"
            elif mode == "GraphRAG":
                endpoint = "/chat/graphrag"

            with st.spinner("Retrieving evidence and generating answers..."):
                response = _api_post(endpoint, payload)

            if mode == "RAG":
                _render_rag_result(response)
            elif mode == "GraphRAG":
                _render_graphrag_result(response)
            else:
                rag_payload = response.get("rag", {})
                graphrag_payload = response.get("graphrag", {})

                col_rag, col_graphrag = st.columns(2)

                with col_rag:
                    st.header("RAG")
                    _render_rag_result(rag_payload)

                with col_graphrag:
                    st.header("GraphRAG")
                    _render_graphrag_result(graphrag_payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8") if exc.fp else ""
            st.error(f"HTTP error from API: {exc.code} {exc.reason}\n{detail}")
        except urllib.error.URLError as exc:
            st.error(f"Could not reach API at {API_BASE_URL}: {exc.reason}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
