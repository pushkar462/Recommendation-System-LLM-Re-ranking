"""
FastAPI backend for the two-stage recommendation system.
Endpoints:
  GET  /users                        - list available users
  GET  /recommend/{user_id}          - full two-stage pipeline
  GET  /candidates/{user_id}         - Stage 1 only (MF candidates)
  GET  /history/{user_id}            - user rating history
  POST /rerank                        - Stage 2 only (re-rank given candidates)
"""

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from dataset import load_dataset, get_user_ids
from matrix_factorization import MatrixFactorizationRecommender
from llm_reranker import llm_rerank
from local_ltr_reranker import LearningToRankReranker

app = FastAPI(title="Two-Stage RecSys API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: train MF model ────────────────────────────────────────────────────

mf_movie = MatrixFactorizationRecommender(n_factors=10)
mf_product = MatrixFactorizationRecommender(n_factors=8)
anthropic_api_key_override: Optional[str] = None
ltr_movie: Optional[LearningToRankReranker] = None
ltr_product: Optional[LearningToRankReranker] = None

@app.on_event("startup")
def startup():
    ratings_m, items_m = load_dataset("movies")
    mf_movie.fit(ratings_m, items_m)
    global ltr_movie
    ltr_movie = LearningToRankReranker.train_from_ratings(
        ratings=ratings_m, items=items_m, mf=mf_movie
    )

    ratings_p, items_p = load_dataset("products")
    mf_product.fit(ratings_p, items_p)
    global ltr_product
    ltr_product = LearningToRankReranker.train_from_ratings(
        ratings=ratings_p, items=items_p, mf=mf_product
    )
    print("✓ Both MF models ready.")

def get_mf(dataset: str) -> MatrixFactorizationRecommender:
    if dataset == "products":
        return mf_product
    return mf_movie


def get_ltr(dataset: str) -> LearningToRankReranker:
    reranker = ltr_product if dataset == "products" else ltr_movie
    if reranker is None:
        raise HTTPException(500, "LTR reranker not initialised yet")
    return reranker


# ── Endpoints ─────────────────────────────────────────────────────────────────

class AnthropicKeyRequest(BaseModel):
    api_key: str


@app.get("/health")
def health():
    configured = bool(anthropic_api_key_override) or bool(os.environ.get("ANTHROPIC_API_KEY"))
    return {"ok": True, "anthropic_configured": configured}


@app.post("/config/anthropic_key")
def set_anthropic_key(req: AnthropicKeyRequest):
    """
    Store ANTHROPIC key in-memory for this server process only.
    This avoids needing terminal exports and keeps the key off the browser-to-Anthropic path.
    """
    global anthropic_api_key_override
    key = (req.api_key or "").strip()
    if not key:
        raise HTTPException(400, "api_key is required")
    anthropic_api_key_override = key
    return {"ok": True, "anthropic_configured": True}


@app.get("/users")
def list_users():
    return {"users": get_user_ids()}


@app.get("/history/{user_id}")
def user_history(
    user_id: str,
    dataset: str = Query("movies", enum=["movies", "products"]),
):
    mf = get_mf(dataset)
    history = mf.get_user_history(user_id)
    if not history:
        raise HTTPException(404, f"User '{user_id}' not found or has no history")
    return {"user_id": user_id, "history": history}


@app.get("/candidates/{user_id}")
def stage1_candidates(
    user_id: str,
    top_k: int = Query(20, ge=5, le=50),
    dataset: str = Query("movies", enum=["movies", "products"]),
):
    """Stage 1: collaborative filtering candidates."""
    mf = get_mf(dataset)
    candidates = mf.get_candidates(user_id, top_k=top_k)
    return {
        "user_id": user_id,
        "stage": "1_collaborative_filtering",
        "n_candidates": len(candidates),
        "candidates": candidates,
    }


class RerankRequest(BaseModel):
    user_id: str
    candidates: List[dict]
    user_history: List[dict]
    user_context: Optional[str] = None
    top_k: int = 10
    dataset: str = "movies"


@app.post("/rerank")
async def stage2_rerank(req: RerankRequest):
    """Stage 2: LTR model by default, Claude optional."""
    if anthropic_api_key_override or os.environ.get("ANTHROPIC_API_KEY"):
        result = await llm_rerank(
            candidates=req.candidates,
            user_history=req.user_history,
            user_context=req.user_context,
            top_k=req.top_k,
            api_key=anthropic_api_key_override,
        )
        return {"user_id": req.user_id, "stage": "2_llm_reranking", **result}

    ltr = get_ltr(req.dataset)
    result = ltr.rerank(
        candidates=req.candidates,
        user_history=req.user_history,
        user_context=req.user_context,
        top_k=req.top_k,
    )
    return {"user_id": req.user_id, "stage": "2_ltr_reranking", **result}


@app.get("/recommend/{user_id}")
async def full_pipeline(
    user_id: str,
    top_k: int = Query(10, ge=3, le=20),
    candidates_k: int = Query(20, ge=10, le=50),
    user_context: Optional[str] = Query(None),
    dataset: str = Query("movies", enum=["movies", "products"]),
):
    """Full two-stage pipeline: MF → (LTR by default, Claude optional)."""
    mf = get_mf(dataset)

    # Stage 1
    candidates = mf.get_candidates(user_id, top_k=candidates_k)
    history = mf.get_user_history(user_id)

    # Stage 2
    if anthropic_api_key_override or os.environ.get("ANTHROPIC_API_KEY"):
        result = await llm_rerank(
            candidates=candidates,
            user_history=history,
            user_context=user_context,
            top_k=top_k,
            api_key=anthropic_api_key_override,
        )
        pipeline = "matrix_factorization → llm_reranking"
    else:
        ltr = get_ltr(dataset)
        result = ltr.rerank(
            candidates=candidates,
            user_history=history,
            user_context=user_context,
            top_k=top_k,
        )
        pipeline = "matrix_factorization → ltr_reranking"

    return {
        "user_id": user_id,
        "dataset": dataset,
        "pipeline": pipeline,
        "stage1_candidates": len(candidates),
        "stage2_top_k": top_k,
        "user_context": user_context,
        "rerank_summary": result.get("rerank_summary"),
        "recommendations": result.get("ranked_items", []),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
