"""API REST: carga un índice persistido una sola vez al iniciar."""
from contextlib import asynccontextmanager
import os
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from retrieval import INDEX_DIR, SearchIndex


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str
    model: Literal['bm25', 'tfidf']
    top_k: StrictInt = Field(gt=0)

    @field_validator('query')
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("query no puede estar vacía")
        return value


class SearchResult(BaseModel):
    rank: int
    document_id: str
    score: float = Field(allow_inf_nan=False)
    title: str
    snippet: str


class SearchResponse(BaseModel):
    model: Literal['bm25', 'tfidf']
    results: list[SearchResult]


@asynccontextmanager
async def lifespan(application):
    location = os.environ.get('INDEX_DIR', INDEX_DIR)
    application.state.index = SearchIndex.load(location)
    yield


app = FastAPI(title="NFCorpus · Jhonn Saeteros", version="1.0.0", lifespan=lifespan)


@app.get('/health')
def health():
    return {"status": "ok", "documents": app.state.index.N, "models": ["tfidf", "bm25"]}


@app.post('/search', response_model=SearchResponse)
def search(body: SearchRequest):
    return {"model": body.model, "results": app.state.index.search(body.query, body.model, body.top_k)}

