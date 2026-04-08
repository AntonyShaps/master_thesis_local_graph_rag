## Local Graph RAG Setup

### 1) Start Neo4j

```bash
docker compose up -d
```

Neo4j defaults from `docker-compose.yml`:

- Bolt: `bolt://localhost:7687`
- Browser: `http://localhost:7474`
- Auth: `neo4j/testpassword`

### 2) Install Python dependencies

```bash
uv sync
```

### 3) Build vector index from markdown reports

```bash
python scripts/to_vector_index.py build --replace-existing
```

This script:

- parses `data/markdown/*.md` by `<page_number>...</page_number>` tags,
- builds page chunks with overlap (`50%` previous + current + `50%` next),
- embeds with `sentence-transformers/all-mpnet-base-v2`,
- stores chunks + embeddings in Neo4j.

### 4) Extract entities + relationships (local Mistral)

```bash
python scripts/entity_extraction.py --companies nvidia google meta
```

Useful options:

- `--max-pages 5` for a quick smoke run.
- `--max-chars-per-unit 3200` to control per-call text unit size.
- `--output-dir data/graph_extraction` to change output location.
- `--resume` to continue from existing `records_raw.jsonl` progress.
- `--no-retry-failed` (with `--resume`) to skip previously failed units.
- `--checkpoint-every-n-units 10` to persist progress during long runs.
- `--no-stop-on-fatal-model-error` if you want the batch to continue after a fatal CUDA error.

For each company, outputs are written to `data/graph_extraction/<company>/`:

- `entities.json`
- `relationships.json`
- `graph_records.json`
- `records_raw.jsonl` (raw model output + parsed records per page/unit)
- `run_summary.json`

### 5) Aggregate + summarize graph and load to Neo4j

Build summarized, company-scoped graph artifacts from existing extraction output:

```bash
uv run python scripts/to_graph_index.py build --companies google meta nvidia
```

This step:

- merges repeated entity mentions by company + canonical entity name,
- aggregates relationship instances by company + (source, target),
- summarizes multi-description entities/relationships with local Mistral,
- writes artifacts to `data/graph_index/<company>/`.

Load summarized graph artifacts into Neo4j:

```bash
uv run python scripts/to_graph_index.py load --companies google meta nvidia --replace-existing
```

Or run both in one command:

```bash
uv run python scripts/to_graph_index.py build-load --companies google meta nvidia --replace-existing
```

### 6) Run FastAPI backend (Mistral + RAG)

```bash
uv run uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Default model path:

- `models/Mistral-7B-Instruct-v0.3`

### 7) Run Streamlit frontend

```bash
uv run streamlit run streamlit_app.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

## Environment Variables

- `NEO4J_URI` (default `bolt://localhost:7687`)
- `NEO4J_AUTH` (default `neo4j/testpassword`)
- `NEO4J_USER` / `NEO4J_PASSWORD` (override `NEO4J_AUTH`)
- `NEO4J_DATABASE` (default `neo4j`)
- `EMBEDDING_MODEL_NAME` (default `sentence-transformers/all-mpnet-base-v2`)
- `MISTRAL_MODEL_DIR` (default `models/Mistral-7B-Instruct-v0.3`)
- `RAG_MAX_CHARS_PER_CHUNK` (default `2200`)
- `RAG_API_URL` for Streamlit (default `http://localhost:8000`)
