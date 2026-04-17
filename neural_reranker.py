"""
Stage 2: Neural MLP Re-ranker.

Fully replaces both:
  - local_ltr_reranker.py  (LightGBM LambdaRank)
  - llm_reranker.py        (Claude Sonnet)

Architecture:
  - sentence-transformers encodes user_context → 384d vector
  - cosine similarity between context and item text = context_score
  - MLP takes 9 features → relevance score in 0..1
  - Trained with pairwise BPR loss
  - MMR post-processing for diversity

Features used:
  1. cf_score          : two-tower dot-product score (normalised 0..1)
  2. popularity        : log-normalised rating count
  3. freshness         : release year normalised 0..1
  4. genre_id          : genre as normalised integer
  5. similarity_to_likes : genre + tag overlap with user's top-rated items
  6. context_score     : cosine sim between user_context and item text
  7. tag_overlap       : jaccard overlap of item tags with liked item tags
  8. avg_item_rating   : mean rating for this item (normalised)
  9. bias              : constant 1.0

No API key needed. Fully offline.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Optional sentence-transformers (graceful fallback if not installed)
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
    print("sentence-transformers not found. Install with: pip install sentence-transformers")
    print("Falling back to keyword-based context scoring.")

# Render free tier (~512Mi): set RERANKER_USE_KEYWORDS_ONLY=1 to skip loading SBERT
if os.environ.get("RERANKER_USE_KEYWORDS_ONLY", "").lower() in ("1", "true", "yes"):
    _SBERT_AVAILABLE = False
    print("RERANKER_USE_KEYWORDS_ONLY=1 — keyword context scoring only (saves RAM).")


SBERT_MODEL_NAME = "all-MiniLM-L6-v2"   # 80 MB, fast, good quality

# Lazy-loaded singleton so we don't reload on every request
_sbert_model: Optional[object] = None


def _get_sbert():
    global _sbert_model
    if _sbert_model is None and _SBERT_AVAILABLE:
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    return _sbert_model


# ---------------------------------------------------------------------------
# Context scoring
# ---------------------------------------------------------------------------

# Genre keyword map used as fallback when sentence-transformers not available
_GENRE_KEYWORD_MAP = {
    "crime"    : {"criminal", "crime", "murder", "heist", "mafia", "gangster", "detective", "cops"},
    "thriller" : {"thriller", "suspense", "tense", "dark", "serial", "killer", "stalker"},
    "horror"   : {"scary", "horror", "fear", "ghost", "monster", "evil", "nightmare"},
    "sci-fi"   : {"sci-fi", "space", "future", "robot", "alien", "technology", "cyber"},
    "drama"    : {"drama", "emotional", "touching", "deep", "powerful", "life"},
    "action"   : {"action", "fight", "explosion", "adventure", "superhero", "battle"},
    "romance"  : {"romance", "love", "romantic", "relationship", "dating", "couple"},
    "animation": {"cartoon", "animated", "anime", "family", "kids", "pixar"},
    "mystery"  : {"mystery", "whodunit", "detective", "clue", "investigation"},
    "fantasy"  : {"fantasy", "magic", "wizard", "dragon", "epic", "quest"},
}

def _extract_intent_genres(user_context: Optional[str], known_genres: set[str]) -> list[str]:
    """
    Try to detect if the user is explicitly asking for a genre (e.g. "horror",
    "crime thriller", "sci fi"). This is used to apply a strong, *generic*
    boost so the top results match the user's stated intent.
    """
    if not user_context:
        return []
    ctx = user_context.lower().replace("-", " ").replace("_", " ")
    tokens = set(ctx.split())

    # Direct matches on known genres
    intents: list[str] = []
    for g in sorted(known_genres):
        g_norm = g.lower().replace("-", " ").replace("_", " ")
        g_tokens = set(g_norm.split())
        if g_tokens and g_tokens.issubset(tokens):
            intents.append(g)

    # Keyword-map matches (maps phrases like "mafia" -> crime)
    for g, keywords in _GENRE_KEYWORD_MAP.items():
        if tokens & keywords:
            # Map to a dataset genre casing if possible; else keep keyword-map genre.
            match = next((kg for kg in known_genres if kg.lower() == g), g)
            if match not in intents:
                intents.append(match)

    return intents[:2]  # keep it simple; multiple intents still supported


def _keyword_context_score(user_context: str, item: dict) -> float:
    """Fallback: keyword-based context scoring."""
    if not user_context:
        return 0.5
    ctx_tokens = set(user_context.lower().split())
    genre      = (item.get("genre") or "").lower()
    tags       = set(t.lower() for t in (item.get("tags") or []))

    best = 0.0
    for g, keywords in _GENRE_KEYWORD_MAP.items():
        overlap = len(ctx_tokens & keywords)
        if overlap > 0:
            genre_match = 1.0 if genre == g else 0.0
            tag_match   = len(tags & keywords) / max(1, len(keywords))
            score       = overlap * 0.3 + genre_match * 0.5 + tag_match * 0.2
            best        = max(best, min(1.0, score))
    return best


def context_score(user_context: str, item: dict) -> float:
    """
    Compute semantic similarity between user_context and item.
    Uses sentence-transformers if available, otherwise keyword fallback.
    """
    if not user_context:
        return 0.5

    sbert = _get_sbert()
    if sbert is None:
        return _keyword_context_score(user_context, item)

    item_text = " ".join([
        item.get("title", ""),
        item.get("genre", ""),
        " ".join(item.get("tags") or []),
    ]).strip()

    vecs     = sbert.encode([user_context, item_text], convert_to_numpy=True)
    v1, v2   = vecs[0], vecs[1]
    cos_sim  = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
    return float((cos_sim + 1.0) / 2.0)   # shift -1..1 → 0..1


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _norm_year(year, lo: int = 1970, hi: int = 2026) -> float:
    try:
        y = max(lo, min(hi, int(year)))
    except Exception:
        y = 2000
    return (y - lo) / (hi - lo)


def _similarity_to_likes(user_history: List[dict], item: dict) -> float:
    liked = [h for h in user_history if float(h.get("rating", 0)) >= 4.0][:10]
    if not liked:
        return 0.0
    genre  = (item.get("genre") or "").lower()
    tags   = set(t.lower() for t in (item.get("tags") or []))
    score  = 0.0
    for h in liked:
        if (h.get("genre") or "").lower() == genre and genre:
            score += 0.6
        htags = set(t.lower() for t in (h.get("tags") or []))
        if tags and htags:
            score += 0.4 * (len(tags & htags) / max(1, len(tags | htags)))
    return min(1.0, score / max(1, len(liked)))


def _tag_overlap_with_liked(user_history: List[dict], item: dict) -> float:
    liked     = [h for h in user_history if float(h.get("rating", 0)) >= 4.0]
    liked_tags = set(t.lower() for h in liked for t in (h.get("tags") or []))
    item_tags  = set(t.lower() for t in (item.get("tags") or []))
    if not liked_tags or not item_tags:
        return 0.0
    return len(liked_tags & item_tags) / max(1, len(liked_tags | item_tags))


def build_feature_vector(
    item: dict,
    user_history: List[dict],
    genre_to_id: Dict[str, int],
    n_genres: int,
    user_context: Optional[str] = None,
    fast_context: bool = False,
) -> List[float]:
    """Build the 9-dimensional feature vector for one (user, item) pair."""
    cf      = float(item.get("predicted_rating", 3.0)) / 5.0
    pop     = float(item.get("popularity", 0.0))
    fresh   = _norm_year(item.get("year", 2000))
    genre   = (item.get("genre") or "unknown").lower()
    gid     = float(genre_to_id.get(genre, 0)) / max(1, n_genres - 1)
    sim     = _similarity_to_likes(user_history, item)
    if not user_context:
        ctx = 0.5
    elif fast_context:
        # Training-time fast path: avoid SBERT calls inside Dataset __getitem__.
        ctx = _keyword_context_score(user_context, item)
    else:
        ctx = context_score(user_context, item)
    tag_ov  = _tag_overlap_with_liked(user_history, item)

    # avg_item_rating from item metadata if available, else fallback to cf
    avg_ir  = float(item.get("avg_item_rating", cf * 5.0)) / 5.0

    return [cf, pop, fresh, gid, sim, ctx, tag_ov, avg_ir, 1.0]


# ---------------------------------------------------------------------------
# Neural MLP model
# ---------------------------------------------------------------------------

class RerankerMLP(nn.Module):
    """
    Simple 3-layer MLP that maps a feature vector → relevance score in (0, 1).
    Small enough to train on CPU in under a minute on synthetic data.
    """

    def __init__(self, input_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class PairwiseRerankerDataset(Dataset):
    def __init__(
        self,
        ratings: List[dict],
        items_by_id: Dict[str, dict],
        genre_to_id: Dict[str, int],
        n_genres: int,
        neg_samples: int = 3,
        max_pairs: int = 20000,
        seed: int = 42,
    ):
        self.rng         = random.Random(seed)
        self.items_by_id = items_by_id
        self.genre_to_id = genre_to_id
        self.n_genres    = n_genres
        self.neg_samples = neg_samples
        self.all_item_ids = list(items_by_id.keys())

        # Build per-user history and positive/negative pairs
        by_user: Dict[str, Dict[str, float]] = {}
        for r in ratings:
            by_user.setdefault(r["user_id"], {})[r["item_id"]] = float(r.get("rating", 3.0))

        self.pairs: List[Tuple[str, str, List[dict]]] = []   # (uid, pos_iid, user_history)
        for uid, rated in by_user.items():
            history = [
                {"item_id": iid, "rating": rt, **items_by_id.get(iid, {})}
                for iid, rt in rated.items()
            ]
            positives = [iid for iid, rt in rated.items() if rt >= 4.0 and iid in items_by_id]
            for iid in positives:
                self.pairs.append((uid, iid, history))

        # Cap training pairs for reasonable CPU time, but sample (don't slice) so we
        # don't systematically drop later users/items.
        if max_pairs and len(self.pairs) > max_pairs:
            self.pairs = self.rng.sample(self.pairs, max_pairs)

    def __len__(self):
        return len(self.pairs) * self.neg_samples

    def _sample_context(self, pos_item: dict) -> Optional[str]:
        """
        Generate a lightweight training context so the model learns how to use
        the context feature at inference time (instead of seeing it as constant).
        """
        if self.rng.random() < 0.35:
            return None
        genre = (pos_item.get("genre") or "").strip().lower()
        tags  = [t for t in (pos_item.get("tags") or []) if isinstance(t, str)]
        # Prefer a strong genre keyword; fall back to a representative tag.
        if genre:
            return genre
        if tags:
            return self.rng.choice(tags)
        return None

    def __getitem__(self, idx):
        uid, pos_iid, history = self.pairs[idx // self.neg_samples]
        seen = {p[1] for p in self.pairs if p[0] == uid}

        neg_iid = self.rng.choice(self.all_item_ids)
        for _ in range(20):
            if neg_iid not in seen:
                break
            neg_iid = self.rng.choice(self.all_item_ids)

        pos_item = {**self.items_by_id.get(pos_iid, {}), "predicted_rating": 4.5}
        neg_item = {**self.items_by_id.get(neg_iid, {}), "predicted_rating": 2.5}

        user_context = self._sample_context(pos_item)
        pos_feat = build_feature_vector(
            pos_item, history, self.genre_to_id, self.n_genres, user_context, fast_context=True
        )
        neg_feat = build_feature_vector(
            neg_item, history, self.genre_to_id, self.n_genres, user_context, fast_context=True
        )

        return (
            torch.tensor(pos_feat, dtype=torch.float32),
            torch.tensor(neg_feat, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Main reranker class
# ---------------------------------------------------------------------------

class NeuralReranker:
    """
    Fully offline neural re-ranker.
    Drop-in replacement for both LTR and Claude Sonnet paths.

    Public API matches LearningToRankReranker:
        rerank(candidates, user_history, user_context, top_k) → dict
    """

    def __init__(
        self,
        model: RerankerMLP,
        genre_to_id: Dict[str, int],
        n_genres: int,
        popularity: Dict[str, float],
    ):
        self.model       = model
        self.genre_to_id = genre_to_id
        self.n_genres    = n_genres
        self.popularity  = popularity

    @staticmethod
    def train(
        ratings: List[dict],
        items: List[dict],
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
        max_pairs: int = 5000,
    ) -> "NeuralReranker":
        torch.manual_seed(seed)
        random.seed(seed)

        items_by_id = {it["item_id"]: it for it in items}
        genres      = sorted({(it.get("genre") or "unknown").lower() for it in items})
        genre_to_id = {g: i for i, g in enumerate(genres)}
        n_genres    = len(genres)

        # Popularity map
        pop_counts: Dict[str, int] = {}
        n_users = len({r["user_id"] for r in ratings}) or 1
        for r in ratings:
            pop_counts[r["item_id"]] = pop_counts.get(r["item_id"], 0) + 1
        popularity = {
            iid: float(np.log1p(cnt)) / float(np.log1p(n_users))
            for iid, cnt in pop_counts.items()
        }
        # Inject popularity into item metadata
        for it in items:
            it["popularity"]      = popularity.get(it["item_id"], 0.0)
            it["avg_item_rating"] = 3.0   # will be overwritten below

        # Compute per-item average rating
        sum_r: Dict[str, float] = {}
        cnt_r: Dict[str, int]   = {}
        for r in ratings:
            iid = r["item_id"]
            sum_r[iid] = sum_r.get(iid, 0.0) + float(r.get("rating", 3.0))
            cnt_r[iid] = cnt_r.get(iid, 0) + 1
        for it in items:
            iid = it["item_id"]
            it["avg_item_rating"] = sum_r.get(iid, 3.0) / max(1, cnt_r.get(iid, 1))
        for it in items:
            items_by_id[it["item_id"]] = it

        ds     = PairwiseRerankerDataset(
            ratings,
            items_by_id,
            genre_to_id,
            n_genres,
            seed=seed,
            max_pairs=max_pairs,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

        model = RerankerMLP(input_dim=9)
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        print(f"Training neural re-ranker for {epochs} epochs on {len(ds)} pairs...")
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            for pos_feat, neg_feat in loader:
                pos_s = model(pos_feat)
                neg_s = model(neg_feat)
                loss  = _bpr_loss(pos_s, neg_s)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total += loss.item()
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} | loss: {total / max(1, len(loader)):.4f}")

        model.eval()
        print("✓ Neural re-ranker trained.")
        return NeuralReranker(model, genre_to_id, n_genres, popularity)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def _score_candidates(
        self,
        candidates: List[dict],
        user_history: List[dict],
        user_context: Optional[str],
    ) -> np.ndarray:
        feats = []
        for c in candidates:
            item = {**c, "popularity": self.popularity.get(c.get("item_id", ""), 0.0)}
            fv   = build_feature_vector(item, user_history, self.genre_to_id, self.n_genres, user_context)
            feats.append(fv)

        x = torch.tensor(feats, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(x).numpy()
        return scores

    @staticmethod
    def _mmr(
        candidates: List[dict],
        scores: np.ndarray,
        top_k: int,
        lambda_mmr: float = 0.7,
        intent_genres: Optional[set[str]] = None,
    ) -> tuple[list[int], np.ndarray]:
        """
        Maximal Marginal Relevance.
        lambda_mmr close to 1.0 = prioritise relevance.
        lambda_mmr close to 0.0 = prioritise diversity.
        """
        selected_idx: list[int] = []
        selected_mmr_scores: list[float] = []
        remaining_idx = list(range(len(candidates)))

        def _tags(it: dict) -> set[str]:
            return {t.lower() for t in (it.get("tags") or []) if isinstance(t, str)}

        def _similarity(i: int, selected: list[int]) -> float:
            """
            Soft similarity in [0, 1] used for diversity penalty.
            Old code used a hard 0/1 genre match which could push a very relevant
            item too far down (and then the UI score looked "wrong").
            """
            if not selected:
                return 0.0
            gi = (candidates[i].get("genre") or "").lower()
            ti = _tags(candidates[i])
            sims = []
            for j in selected:
                gj = (candidates[j].get("genre") or "").lower()
                tj = _tags(candidates[j])
                # If user explicitly asked for a genre, we *don't* want diversity
                # penalty to push that genre down.
                if intent_genres and gi in intent_genres and gj in intent_genres:
                    genre_sim = 0.0
                else:
                    genre_sim = 1.0 if gi and gi == gj else 0.0
                if ti and tj:
                    tag_sim = len(ti & tj) / max(1, len(ti | tj))
                else:
                    tag_sim = 0.0
                sims.append(0.6 * genre_sim + 0.4 * tag_sim)
            return float(max(sims)) if sims else 0.0

        def _mmr_score(i: int, selected: list[int]) -> float:
            rel = float(scores[i])
            sim = _similarity(i, selected)
            return float(lambda_mmr * rel - (1.0 - lambda_mmr) * sim)

        while len(selected_idx) < top_k and remaining_idx:
            if not selected_idx:
                best = max(remaining_idx, key=lambda i: scores[i])
                best_score = float(scores[best])
            else:
                best = max(remaining_idx, key=lambda i: _mmr_score(i, selected_idx))
                best_score = _mmr_score(best, selected_idx)

            selected_idx.append(best)
            selected_mmr_scores.append(best_score)
            remaining_idx.remove(best)

        return selected_idx, np.array(selected_mmr_scores, dtype=np.float32)

    @staticmethod
    def _generate_reason(item: dict, user_history: List[dict], score: float) -> str:
        liked = [h for h in user_history if float(h.get("rating", 0)) >= 4.0]
        genre_matches = [h for h in liked if h.get("genre") == item.get("genre")]
        tag_matches   = [
            h for h in liked
            if set(h.get("tags") or []) & set(item.get("tags") or [])
        ]
        if genre_matches:
            return (
                f"Matches your taste for {item.get('genre')} — "
                f"similar to {genre_matches[0].get('title', 'titles you enjoyed')}."
            )
        if tag_matches:
            common = set(tag_matches[0].get("tags") or []) & set(item.get("tags") or [])
            return (
                f"Shares themes ({', '.join(list(common)[:2])}) with "
                f"{tag_matches[0].get('title', 'titles you enjoyed')}."
            )
        return f"Highly relevant {item.get('genre', '')} title based on your viewing profile."

    def rerank(
        self,
        candidates: List[dict],
        user_history: List[dict],
        user_context: Optional[str] = None,
        top_k: int = 10,
        lambda_mmr: float = 0.7,
    ) -> dict:
        """
        Re-rank candidates.

        Args:
            candidates   : Stage 1 output (two-tower or SVD candidates)
            user_history : User's rated items
            user_context : Free-text context e.g. "I want something dark and gritty"
            top_k        : Number of final results
            lambda_mmr   : Diversity control (0.7 = balanced, 1.0 = pure relevance)

        Returns:
            dict with "ranked_items" and "rerank_summary"
        """
        if not candidates:
            return {"ranked_items": [], "rerank_summary": "No candidates."}

        # Never recommend already-rated items.
        seen_ids = {h.get("item_id") for h in (user_history or []) if h.get("item_id")}
        candidates = [c for c in candidates if c.get("item_id") not in seen_ids]
        if not candidates:
            return {"ranked_items": [], "rerank_summary": "No unseen candidates."}

        scores_model = self._score_candidates(candidates, user_history, user_context)

        # Detect explicit genre intent from the prompt.
        known_genres = {g.lower() for g in self.genre_to_id.keys()}
        intents = _extract_intent_genres(user_context, known_genres)
        intent_set = set(intents)

        # Blend in explicit context similarity so the prompt reliably influences rank.
        if user_context:
            scores_ctx = np.array([context_score(user_context, c) for c in candidates], dtype=np.float32)
            # Model outputs are already (0..1). Context is (0..1). Blend them.
            scores = 0.55 * scores_model + 0.45 * scores_ctx
        else:
            scores = scores_model

        # Strong generic boost when a user explicitly asks for a genre.
        # This avoids "horror" prompts returning mostly non-horror at the top.
        if intent_set:
            genres = np.array([(c.get("genre") or "").lower() for c in candidates], dtype=object)
            is_intent = np.array([g in intent_set for g in genres], dtype=np.float32)
            # Boost intent-genre items, and lightly demote others.
            scores = scores + 0.85 * is_intent - 0.10 * (1.0 - is_intent)

            # When intent is present, bias towards relevance (less diversity penalty).
            lambda_mmr = max(lambda_mmr, 0.85)

        # Rank-normalise scores for stable 0..1 UI display
        if scores.max() - scores.min() > 1e-6:
            scores_ui = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores_ui = np.full_like(scores, 0.5)

        # Sort by raw score first
        order      = np.argsort(-scores)
        candidates = [candidates[i] for i in order]
        scores_ui  = scores_ui[order]

        # MMR diversity pass
        ranked_idx, ranked_mmr = self._mmr(
            candidates,
            scores_ui,
            top_k=top_k,
            lambda_mmr=lambda_mmr,
            intent_genres=intent_set if intent_set else None,
        )

        # Normalise the *final* MMR scores for the UI so score == ordering signal.
        if ranked_mmr.size and float(ranked_mmr.max() - ranked_mmr.min()) > 1e-6:
            ranked_score_ui = (ranked_mmr - ranked_mmr.min()) / (ranked_mmr.max() - ranked_mmr.min())
        else:
            ranked_score_ui = np.full_like(ranked_mmr, 0.5, dtype=np.float32)

        ranked_items = []
        for pos, idx in enumerate(ranked_idx):
            item = candidates[idx]
            s = float(ranked_score_ui[pos]) if pos < len(ranked_score_ui) else 0.5
            ranked_items.append({
                **item,
                "rank"  : pos + 1,
                # Keep full precision; UI can format. Rounding here can collapse
                # small but meaningful differences into 0.00.
                "score" : s,
                "reason": self._generate_reason(item, user_history, s),
            })

        summary = (
            f"Re-ranked {len(candidates)} candidates using neural MLP + "
            f"{'sentence-transformers' if _SBERT_AVAILABLE else 'keyword'} context scoring + MMR diversity."
        )
        if user_context:
            summary += f" Context: '{user_context}'."

        return {"ranked_items": ranked_items, "rerank_summary": summary}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "genre_to_id": self.genre_to_id,
            "n_genres"   : self.n_genres,
            "popularity" : self.popularity,
            "input_dim"  : 9,
        }, path)
        print(f"✓ Neural re-ranker saved to {path}")

    @classmethod
    def load(cls, path: str) -> "NeuralReranker":
        data  = torch.load(path, map_location="cpu")
        model = RerankerMLP(input_dim=data.get("input_dim", 9))
        model.load_state_dict(data["model_state"])
        model.eval()
        return cls(model, data["genre_to_id"], data["n_genres"], data["popularity"])
