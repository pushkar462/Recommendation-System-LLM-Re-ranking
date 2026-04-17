"""
FastAPI backend — updated to use:
  Stage 1 : Two-Tower Neural Retrieval  (two_tower.py)
  Stage 2 : Neural MLP Re-ranker        (neural_reranker.py)
  Data    : MovieLens-20M / 1M / 100K   (movielens_dataset.py)

No Anthropic API key needed. Fully offline.

Endpoints (same as before — frontend unchanged):
  GET  /users
  GET  /history/{user_id}
  GET  /candidates/{user_id}
  GET  /recommend/{user_id}
  POST /rerank
  GET  /health

Startup behaviour:
  - Checks for a saved model checkpoint in ./checkpoints/
  - If found: loads it instantly (fast)
  - If not found: trains from scratch and saves checkpoint

Training time estimates:
  synthetic  : ~10 seconds
  ml-100k    : ~2 minutes
  ml-1m      : ~10 minutes
  ml-20m     : ~60 minutes (use sample_users=20000 to cap at ~15 minutes)
"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from movielens_dataset import load_dataset, get_user_ids
from two_tower import TwoTowerRetriever, train_two_tower
from neural_reranker import NeuralReranker

app = FastAPI(title="Two-Stage RecSys API — Neural Edition", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMO_HTML_PATH = Path(__file__).parent / "demo.html"

# ---------------------------------------------------------------------------
# Globals (populated on startup)
# ---------------------------------------------------------------------------

retriever: Optional[TwoTowerRetriever] = None
reranker:  Optional[NeuralReranker]    = None
_all_ratings: List[dict] = []
_all_items:   List[dict] = []

# ---------------------------------------------------------------------------
# Dataset config — change these to switch datasets
# ---------------------------------------------------------------------------

DATASET_VARIANT = "ml-100k"   # was "auto" which picks ml-1m (much larger)
SAMPLE_USERS    = 500         # cap users loaded into memory
                              # None = use all users

CHECKPOINT_DIR = Path("./checkpoints")

# Include dataset config in checkpoint names so deployments don't accidentally
# reuse checkpoints trained on a different dataset/user-id schema.
def _ckpt_suffix() -> str:
    su = "all" if SAMPLE_USERS is None else str(SAMPLE_USERS)
    return f"{DATASET_VARIANT}__users_{su}"


RETRIEVER_CHECKPOINT = CHECKPOINT_DIR / f"two_tower__{_ckpt_suffix()}.pt"
RERANKER_CHECKPOINT  = CHECKPOINT_DIR / f"neural_reranker__{_ckpt_suffix()}.pt"

# Training epochs — reduce for faster startup, increase for better quality
TWO_TOWER_EPOCHS = 30
RERANKER_EPOCHS  = 10


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    global retriever, reranker, _all_ratings, _all_items

    # ── Load data ─────────────────────────────────────────────────────────
    # On Render, MovieLens files won't exist unless you vendor them into the repo.
    # Fall back to the built-in synthetic dataset so the service can boot.
    print(f"\nLoading dataset (variant={DATASET_VARIANT!r})...")
    try:
        _all_ratings, _all_items = load_dataset(DATASET_VARIANT, sample_users=SAMPLE_USERS)
    except FileNotFoundError as e:
        print(f"⚠ {e}")
        print("⚠ Falling back to synthetic dataset (no external files required).")
        _all_ratings, _all_items = load_dataset("synthetic", sample_users=SAMPLE_USERS)
    print(f"  {len(_all_ratings):,} ratings | {len(_all_items):,} items | "
          f"{len({r['user_id'] for r in _all_ratings}):,} users")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Two-Tower Retriever ───────────────────────────────────────────────
    # Even if a checkpoint exists, validate it matches current dataset IDs.
    # Render-like platforms can persist disk across deploys; if the dataset
    # user-id schema changes (e.g. synthetic 'alice' vs numeric ids), the
    # retriever would silently fall back to cold-start candidates.
    need_train_retriever = not RETRIEVER_CHECKPOINT.exists()
    if not need_train_retriever:
        try:
            print(f"\nLoading two-tower retriever from checkpoint...")
            retriever = TwoTowerRetriever.load(str(RETRIEVER_CHECKPOINT), _all_ratings)

            sample_users = get_user_ids(_all_ratings)[:5]
            missing = [u for u in sample_users if u not in (retriever.user_feats or {})]
            if missing:
                print(
                    "⚠ Two-tower checkpoint doesn't match current dataset "
                    f"(missing users: {missing}). Re-training retriever..."
                )
                need_train_retriever = True
        except Exception as e:
            print(f"⚠ Failed to load retriever checkpoint: {e}. Re-training retriever...")
            need_train_retriever = True

    if need_train_retriever:
        print(f"\nTraining two-tower retriever (epochs={TWO_TOWER_EPOCHS})...")
        retriever = train_two_tower(
            ratings=_all_ratings,
            items=_all_items,
            epochs=TWO_TOWER_EPOCHS,
        )
        retriever.save(str(RETRIEVER_CHECKPOINT))

    # ── Neural Re-ranker ──────────────────────────────────────────────────
    need_train_reranker = not RERANKER_CHECKPOINT.exists()
    if not need_train_reranker:
        try:
            print(f"\nLoading neural re-ranker from checkpoint...")
            reranker = NeuralReranker.load(str(RERANKER_CHECKPOINT))
        except Exception as e:
            print(f"⚠ Failed to load reranker checkpoint: {e}. Re-training reranker...")
            need_train_reranker = True

    if need_train_reranker:
        print(f"\nTraining neural re-ranker (epochs={RERANKER_EPOCHS})...")
        reranker = NeuralReranker.train(
            ratings=_all_ratings,
            items=_all_items,
            epochs=RERANKER_EPOCHS,
        )
        reranker.save(str(RERANKER_CHECKPOINT))

    print("\n✓ Server ready.\n")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_retriever() -> TwoTowerRetriever:
    if retriever is None:
        raise HTTPException(503, "Retriever not ready yet — server is still starting up.")
    return retriever


def _get_reranker() -> NeuralReranker:
    if reranker is None:
        raise HTTPException(503, "Re-ranker not ready yet — server is still starting up.")
    return reranker


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "ok"              : True,
        "retriever_ready" : retriever is not None,
        "reranker_ready"  : reranker is not None,
        "n_ratings"       : len(_all_ratings),
        "n_items"         : len(_all_items),
        "pipeline"        : "two_tower → neural_reranker",
    }


@app.get("/")
def root():
    if not DEMO_HTML_PATH.exists():
        raise HTTPException(404, "demo.html not found in repository root.")
    # Served from the same origin as the API; demo.html uses relative API calls.
    return FileResponse(str(DEMO_HTML_PATH))


@app.get("/users")
def list_users():
    return {"users": get_user_ids(_all_ratings)}


@app.get("/history/{user_id}")
def user_history(user_id: str):
    r = _get_retriever()
    history = r.get_user_history(user_id)
    if not history:
        raise HTTPException(404, f"User '{user_id}' not found or has no history.")
    return {"user_id": user_id, "history": history}


@app.get("/candidates/{user_id}")
def stage1_candidates(
    user_id: str,
    top_k: int = Query(20, ge=5, le=100),
):
    """Stage 1: Two-Tower retrieval candidates."""
    r          = _get_retriever()
    candidates = r.get_candidates(user_id, top_k=top_k)
    return {
        "user_id"     : user_id,
        "stage"       : "1_two_tower_retrieval",
        "n_candidates": len(candidates),
        "candidates"  : candidates,
    }


class RerankRequest(BaseModel):
    user_id     : str
    candidates  : List[dict]
    user_history: List[dict]
    user_context: Optional[str] = None
    top_k       : int = 10
    lambda_mmr  : float = 0.7


@app.post("/rerank")
def stage2_rerank(req: RerankRequest):
    """Stage 2: Neural MLP re-ranking with MMR diversity."""
    rr     = _get_reranker()
    result = rr.rerank(
        candidates   = req.candidates,
        user_history = req.user_history,
        user_context = req.user_context,
        top_k        = req.top_k,
        lambda_mmr   = req.lambda_mmr,
    )
    return {"user_id": req.user_id, "stage": "2_neural_reranking", **result}


@app.get("/recommend/{user_id}")
def full_pipeline(
    user_id     : str,
    top_k       : int   = Query(10, ge=3, le=20),
    candidates_k: int   = Query(20, ge=10, le=100),
    user_context: Optional[str] = Query(None),
    lambda_mmr  : float = Query(0.7, ge=0.0, le=1.0),
):
    """
    Full two-stage pipeline: Two-Tower retrieval → Neural MLP re-ranking.

    Args:
        user_id     : User ID (e.g. 'u1', 'alice')
        top_k       : Final number of recommendations to return
        candidates_k: How many candidates Stage 1 retrieves
        user_context: Free-text mood/context — e.g. "I want something dark and gritty"
        lambda_mmr  : Diversity (0.0 = max diversity, 1.0 = max relevance)

    Example:
        GET /recommend/u1?user_context=criminal+movies&top_k=10
    """
    r = _get_retriever()
    rr = _get_reranker()

    # Stage 1 — two-tower retrieval
    candidates = r.get_candidates(user_id, top_k=candidates_k)
    history    = r.get_user_history(user_id)

    # Stage 2 — neural re-ranking
    result = rr.rerank(
        candidates   = candidates,
        user_history = history,
        user_context = user_context,
        top_k        = top_k,
        lambda_mmr   = lambda_mmr,
    )

    return {
        "user_id"          : user_id,
        "pipeline"         : "two_tower_retrieval → neural_mlp_reranking",
        "stage1_candidates": len(candidates),
        "stage2_top_k"     : top_k,
        "user_context"     : user_context,
        "lambda_mmr"       : lambda_mmr,
        "rerank_summary"   : result.get("rerank_summary"),
        "recommendations"  : result.get("ranked_items", []),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
