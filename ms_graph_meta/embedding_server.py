from __future__ import annotations

import os
import time
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
HOST = os.getenv("EMBEDDING_HOST", "127.0.0.1")
PORT = int(os.getenv("EMBEDDING_PORT", "18011"))

app = FastAPI(title="Local Embedding Server")

model = SentenceTransformer(MODEL_NAME)


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    encoding_format: Literal["float"] | None = "float"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    if isinstance(req.input, str):
        texts = [req.input]
    elif isinstance(req.input, list) and all(isinstance(x, str) for x in req.input):
        texts = req.input
    else:
        raise HTTPException(status_code=400, detail="input must be a string or list of strings")

    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    data = [
        {
            "object": "embedding",
            "index": i,
            "embedding": vector.tolist(),
        }
        for i, vector in enumerate(vectors)
    ]

    token_estimate = sum(max(1, len(text.split())) for text in texts)

    return {
        "object": "list",
        "data": data,
        "model": req.model or MODEL_NAME,
        "usage": {
            "prompt_tokens": token_estimate,
            "total_tokens": token_estimate,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
