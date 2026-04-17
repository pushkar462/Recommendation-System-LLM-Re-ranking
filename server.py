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
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from movielens_dataset import load_dataset, get_user_ids, ensure_ml100k_downloaded
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
_item_popularity: dict = {}
_items_by_id: dict = {}

# ---------------------------------------------------------------------------
# Dataset config — change these to switch datasets
# (Override on Render etc.: DATASET_VARIANT, SAMPLE_USERS)
# ---------------------------------------------------------------------------

def _sample_users_from_env() -> Optional[int]:
    raw = os.environ.get("SAMPLE_USERS")
    if raw is None or str(raw).strip() == "":
        return 500
    s = str(raw).strip().lower()
    if s in ("all", "none", "0"):
        return None
    return int(s)


DATASET_VARIANT = (os.environ.get("DATASET_VARIANT") or "ml-1m").strip() or "ml-1m"
SAMPLE_USERS    = _sample_users_from_env()  # None = use all users (heavy)

CHECKPOINT_DIR = Path("./checkpoints")

# Fixed UI personas (dropdown shows only these names). Real MovieLens users are
# bucketed by genre affinity; each bucket gets one representative user_id for API calls.
PERSONA_SPECS: List[tuple[str, tuple[str, str]]] = [
    ("alice",    ("Drama", "Sci-Fi")),
    ("bob",      ("Action", "Sci-Fi")),
    ("carol",    ("Animation", "Drama")),
    ("dave",     ("Crime", "Thriller")),
    ("eve",      ("Sci-Fi", "Fantasy")),
    ("frank",    ("Romance", "Drama")),
    ("grace",    ("Horror", "Thriller")),
    ("henry",    ("Mystery", "Crime")),
    ("isabella", ("Animation", "Romance")),
    ("jack",     ("Action", "Adventure")),
]


def _user_genre_weights(rlist: List[dict], items_by_id: Dict[str, dict]) -> Dict[str, float]:
    """Per-user genre weights from ratings (rating² on primary item genre)."""
    w: Dict[str, float] = defaultdict(float)
    for rr in rlist:
        it = items_by_id.get(rr.get("item_id"), {})
        g = (it.get("genre") or "Unknown").strip() or "Unknown"
        x = float(rr.get("rating", 3.0))
        w[g] += x * x
    return w


def _persona_score(weights: Dict[str, float], g1: str, g2: str) -> float:
    """Higher = better match to this persona's two target genres."""
    return float(weights.get(g1, 0.0) + weights.get(g2, 0.0))


def _build_persona_user_list(ratings: List[dict], items_by_id: Dict[str, dict]) -> List[dict]:
    """
    Map every loaded user → best-matching persona; return exactly 10 rows (alice..jack).
    Each row: representative real user_id for API routes + label for UI only.

    No ratings are dropped: bucketing is only for choosing which real u#### represents
    each persona in the dropdown.
    """
    by_user: Dict[str, List[dict]] = defaultdict(list)
    for r in ratings:
        by_user[r["user_id"]].append(r)

    all_uids = sorted(by_user.keys())
    if not all_uids:
        return []

    # Non–MovieLens ids (e.g. synthetic dataset: alice, bob, …): map name → persona when possible.
    is_movielens_style = all(re.fullmatch(r"u\d+", uid) for uid in all_uids)
    if not is_movielens_style:
        out = []
        pool = list(all_uids)
        used: set[str] = set()
        for idx, (persona, (g1, g2)) in enumerate(PERSONA_SPECS):
            if persona in by_user and persona not in used:
                uid = persona
            else:
                rem = [u for u in pool if u not in used]
                if rem:
                    uid = rem[idx % len(rem)]
                else:
                    uid = pool[idx % len(pool)]
            used.add(uid)
            label = f"{persona} — {g1} / {g2} fan"
            out.append({
                "persona": persona,
                "user_id": uid,
                "label": label,
                "genres": [g1, g2],
                "in_bucket": 1,
            })
        return out

    user_weights: Dict[str, Dict[str, float]] = {
        uid: _user_genre_weights(by_user[uid], items_by_id) for uid in all_uids
    }

    # Assign each user to argmax persona by affinity score
    buckets: Dict[str, List[str]] = {p: [] for p, _ in PERSONA_SPECS}
    for uid in all_uids:
        wts = user_weights[uid]
        best_p, best_s = PERSONA_SPECS[0][0], -1.0
        for persona, (g1, g2) in PERSONA_SPECS:
            s = _persona_score(wts, g1, g2)
            if s > best_s:
                best_s, best_p = s, persona
        buckets[best_p].append(uid)

    def _rep_for_bucket(persona: str, g1: str, g2: str) -> str:
        members = buckets.get(persona) or []
        if members:
            # Most ratings → richest history for the demo
            return max(members, key=lambda u: len(by_user[u]))
        # Empty bucket: pick real user with highest affinity to this persona anyway
        best_uid, best_sc = all_uids[0], -1.0
        for uid in all_uids:
            s = _persona_score(user_weights[uid], g1, g2)
            if s > best_sc:
                best_sc, best_uid = s, uid
        return best_uid

    out = []
    for persona, (g1, g2) in PERSONA_SPECS:
        uid = _rep_for_bucket(persona, g1, g2)
        label = f"{persona} — {g1} / {g2} fan"
        out.append({
            "persona" : persona,
            "user_id" : uid,
            "label"   : label,
            "genres"  : [g1, g2],
            "in_bucket": len(buckets.get(persona) or []),
        })
    return out


# Include dataset config in checkpoint names so deployments don't accidentally
# reuse checkpoints trained on a different dataset/user-id schema.
def _ckpt_suffix() -> str:
    su = "all" if SAMPLE_USERS is None else str(SAMPLE_USERS)
    return f"{DATASET_VARIANT}__users_{su}"


RETRIEVER_CHECKPOINT = CHECKPOINT_DIR / f"two_tower__{_ckpt_suffix()}.pt"
RERANKER_CHECKPOINT  = CHECKPOINT_DIR / f"neural_reranker__{_ckpt_suffix()}.pt"

# Training epochs — override with TWO_TOWER_EPOCHS / RERANKER_EPOCHS (e.g. Render free tier)
def _env_int_simple(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(str(raw).strip())


TWO_TOWER_EPOCHS = _env_int_simple("TWO_TOWER_EPOCHS", 15)
RERANKER_EPOCHS  = _env_int_simple("RERANKER_EPOCHS", 3)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    global retriever, reranker, _all_ratings, _all_items, _item_popularity, _items_by_id

    # ── Load data ─────────────────────────────────────────────────────────
    # On Render, MovieLens files won't exist unless we download them at runtime.
    # Try to auto-download ML-100K, then fall back to synthetic only if that fails.
    print(f"\nLoading dataset (variant={DATASET_VARIANT!r})...")
    try:
        if DATASET_VARIANT == "ml-100k":
            ensure_ml100k_downloaded()
        _all_ratings, _all_items = load_dataset(DATASET_VARIANT, sample_users=SAMPLE_USERS)
    except FileNotFoundError as e:
        print(f"⚠ {e}")
        print("⚠ Falling back to synthetic dataset (no external files required).")
        _all_ratings, _all_items = load_dataset("synthetic", sample_users=SAMPLE_USERS)
    print(f"  {len(_all_ratings):,} ratings | {len(_all_items):,} items | "
          f"{len({r['user_id'] for r in _all_ratings}):,} users")

    # Build item lookup + popularity for intent-based candidate supplementation.
    _items_by_id = {it["item_id"]: it for it in _all_items if it.get("item_id")}
    pop = {}
    for rr in _all_ratings:
        iid = rr.get("item_id")
        if iid:
            pop[iid] = pop.get(iid, 0) + 1
    _item_popularity = pop

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

            # Validate checkpoint matches current dataset. We need this to be strict,
            # otherwise some users silently fall back to cold-start retrieval.
            current_users = set(get_user_ids(_all_ratings))
            ckpt_users    = set((retriever.user_feats or {}).keys())
            missing_users = sorted(list(current_users - ckpt_users))[:10]

            if current_users != ckpt_users:
                print(
                    "⚠ Two-tower checkpoint doesn't match current dataset "
                    f"(missing_users={missing_users}, "
                    f"ckpt_users={len(ckpt_users):,}, current_users={len(current_users):,}). "
                    "Re-training retriever..."
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
    return {"users": _build_persona_user_list(_all_ratings, _items_by_id)}


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
    history    = r.get_user_history(user_id)
    candidates = r.get_candidates(user_id, top_k=candidates_k)

    # If the user explicitly asked for a genre ("crime", "horror", ...),
    # ensure the candidate pool contains enough items of that genre.
    if user_context:
        ctx = user_context.lower().replace("-", " ").replace("_", " ")
        tokens = set(ctx.split())
        genre_keywords = {
            "crime": {"crime", "criminal", "mafia", "gangster", "heist", "detective", "murder"},
            "horror": {"horror", "scary", "ghost", "monster", "haunted"},
            "thriller": {"thriller", "suspense", "serial", "killer", "dark"},
            "romance": {"romance", "romantic", "love"},
            "sci-fi": {"sci", "scifi", "sci fi", "science", "space", "alien", "robot", "cyber"},
            "mystery": {"mystery", "whodunit", "investigation"},
            "action": {"action", "fight", "explosion", "battle"},
            "comedy": {"comedy", "funny", "humor"},
            "animation": {"animation", "anime", "animated", "pixar"},
            "drama": {"drama"},
        }

        intent_genre = None
        for g, kws in genre_keywords.items():
            if tokens & kws:
                intent_genre = g
                break

        if intent_genre:
            want = 25
            have = sum(1 for c in candidates if (c.get("genre") or "").lower() == intent_genre)
            if have < min(want, candidates_k):
                seen = {h.get("item_id") for h in history if h.get("item_id")}
                existing = {c.get("item_id") for c in candidates if c.get("item_id")}
                # Pick popular items from the requested genre, excluding seen.
                pool = [
                    iid for iid, it in _items_by_id.items()
                    if (it.get("genre") or "").lower() == intent_genre
                    and iid not in seen
                    and iid not in existing
                ]
                pool.sort(key=lambda iid: -_item_popularity.get(iid, 0))
                extra = []
                for iid in pool[: max(0, want - have)]:
                    it = _items_by_id.get(iid, {})
                    extra.append({"item_id": iid, "predicted_rating": 3.0, **it})
                if extra:
                    candidates = candidates + extra

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
