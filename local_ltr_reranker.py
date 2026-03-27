from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import random

import numpy as np
import lightgbm as lgb


def _norm_year(year: Optional[object], lo: int = 1970, hi: int = 2026) -> float:
    if year is None or year == "":
        return 0.5
    try:
        y = int(year)
    except Exception:
        return 0.5
    y = max(lo, min(hi, y))
    return (y - lo) / (hi - lo)


def _context_overlap(context: Optional[str], item: dict) -> float:
    if not context:
        return 0.0
    ctx = [t for t in context.lower().split() if len(t) > 3]
    if not ctx:
        return 0.0
    text = " ".join(
        [
            str(item.get("title", "")),
            str(item.get("genre", "")),
            " ".join(item.get("tags", []) or []),
        ]
    ).lower()
    hits = sum(1 for t in ctx if t in text)
    return min(1.0, hits / max(1, len(ctx)))


def _context_intent_bonus(context: Optional[str], item: dict) -> float:
    """
    Map certain high-signal context words to genres/tags.
    This is intentionally lightweight & deterministic for the no-LLM path.
    Returns a 0..1 bonus.
    """
    if not context:
        return 0.0
    ctx = context.lower()
    g = (item.get("genre") or "").lower()
    tags = set([str(t).lower() for t in (item.get("tags") or [])])

    # Violence / crime intent → Crime/Thriller/Mystery + murder-ish tags
    crime_tokens = {
        "kill",
        "killed",
        "killing",
        "murder",
        "murdered",
        "boyfriend",
        "girlfriend",
        "crime",
        "detective",
        "serial",
        "killer",
        "stalker",
    }
    hit_crime = any(tok in ctx for tok in crime_tokens)
    if hit_crime:
        genre_hit = g in {"crime", "thriller", "mystery", "horror"}
        tag_hit = bool(tags & {"serial-killer", "detective", "fbi", "dark", "violence", "twist", "whodunit"})
        if genre_hit and tag_hit:
            return 1.0
        if genre_hit:
            return 0.75
        if tag_hit:
            return 0.5
        return 0.0

    return 0.0


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    """
    Convert raw model scores to a stable 0..1 scale using rank normalization.
    Avoids the "all ~1.00 after sigmoid" UI issue on tiny synthetic datasets.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    # argsort twice gives ranks; use stable sort to keep deterministic ordering on ties
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(v.size, dtype=float)
    if v.size == 1:
        return np.asarray([0.5], dtype=float)
    return ranks / (v.size - 1.0)


def _similarity_to_likes(user_history: List[dict], item: dict) -> float:
    liked = [h for h in user_history if float(h.get("rating", 0)) >= 4.0][:8]
    if not liked:
        return 0.0
    g = (item.get("genre") or "").lower()
    tags = set([t.lower() for t in (item.get("tags") or [])])
    score = 0.0
    for h in liked:
        if g and (h.get("genre") or "").lower() == g:
            score += 0.6
        htags = set([t.lower() for t in (h.get("tags") or [])])
        if tags and htags:
            score += 0.4 * (len(tags & htags) / max(1, len(tags | htags)))
    return min(1.0, score / max(1, len(liked)))


def _enforce_diversity(items: List[dict], max_per_genre: int = 3) -> List[dict]:
    out: List[dict] = []
    leftovers: List[dict] = []
    counts: Dict[str, int] = {}
    for it in items:
        g = (it.get("genre") or "unknown").lower()
        if counts.get(g, 0) < max_per_genre:
            out.append(it)
            counts[g] = counts.get(g, 0) + 1
        else:
            leftovers.append(it)
    return out + leftovers


@dataclass
class LTRArtifacts:
    model: lgb.LGBMRanker
    genre_to_id: Dict[str, int]
    popularity: Dict[str, float]  # item_id -> 0..1


class LearningToRankReranker:
    """
    LightGBM LambdaRank reranker.
    Trains per-dataset using synthetic ratings + MF CF-scores as a feature.
    """

    def __init__(self, artifacts: LTRArtifacts):
        self.artifacts = artifacts

    @staticmethod
    def train_from_ratings(
        *,
        ratings: List[dict],
        items: List[dict],
        mf,
        seed: int = 7,
    ) -> "LearningToRankReranker":
        rng = random.Random(seed)

        items_by_id = {it["item_id"]: it for it in items}
        genres = sorted({(it.get("genre") or "unknown").lower() for it in items})
        genre_to_id = {g: i for i, g in enumerate(genres)}

        # popularity: fraction of users who rated the item
        user_ids = sorted({r["user_id"] for r in ratings})
        item_ids = sorted({it["item_id"] for it in items})
        pop_counts = {iid: 0 for iid in item_ids}
        seen_pairs = set()
        for r in ratings:
            key = (r["user_id"], r["item_id"])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            if r.get("rating", 0) > 0:
                pop_counts[r["item_id"]] = pop_counts.get(r["item_id"], 0) + 1
        popularity = {iid: (pop_counts.get(iid, 0) / max(1, len(user_ids))) for iid in item_ids}

        def feat_for(user_id: str, item_id: str) -> List[float]:
            it = items_by_id[item_id]
            genre_id = float(genre_to_id.get((it.get("genre") or "unknown").lower(), 0))
            fresh = _norm_year(it.get("year"))
            pop = float(popularity.get(item_id, 0.0))
            # cf_score: use MF score for this user/item (clipped 1..5, normalised 0..1)
            cf_raw = 3.0
            try:
                # get_candidates is expensive; instead compute dot-product for this item only
                if user_id in mf.user_id_map and item_id in mf.item_id_map:
                    u = mf.user_id_map[user_id]
                    i = mf.item_id_map[item_id]
                    cf_raw = float(np.clip((mf.item_factors[i] @ mf.user_factors[u]) + mf.global_mean, 1, 5))
            except Exception:
                cf_raw = 3.0
            cf = cf_raw / 5.0
            return [cf, pop, fresh, genre_id]

        # Build grouped training data (one query per user)
        X: List[List[float]] = []
        y: List[int] = []
        group: List[int] = []

        # Pre-index explicit ratings
        by_user: Dict[str, Dict[str, float]] = {u: {} for u in user_ids}
        for r in ratings:
            by_user.setdefault(r["user_id"], {})[r["item_id"]] = float(r["rating"])

        for u in user_ids:
            rated = by_user.get(u, {})
            # positives: ratings >= 4
            positives = [iid for iid, rt in rated.items() if rt >= 4.0 and iid in items_by_id]
            # negatives: ratings <= 3, plus some unrated sampled as negatives
            negatives = [iid for iid, rt in rated.items() if rt > 0 and rt <= 3.0 and iid in items_by_id]
            unrated = [iid for iid in item_ids if iid not in rated]
            rng.shuffle(unrated)
            negatives += unrated[: min(10, len(unrated))]

            # if user has no positives, skip (not useful for ranking)
            if not positives:
                continue

            user_rows = []
            for iid in positives:
                user_rows.append((1, feat_for(u, iid)))
            for iid in negatives:
                user_rows.append((0, feat_for(u, iid)))

            # shuffle within group
            rng.shuffle(user_rows)
            group.append(len(user_rows))
            for label, feats in user_rows:
                y.append(label)
                X.append(feats)

        X_np = np.asarray(X, dtype=float)
        y_np = np.asarray(y, dtype=int)

        # Train LambdaRank
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
        )
        # genre_id is a categorical-ish signal; LightGBM supports categorical_feature indices
        model.fit(X_np, y_np, group=group, categorical_feature=[3])

        return LearningToRankReranker(
            LTRArtifacts(model=model, genre_to_id=genre_to_id, popularity=popularity)
        )

    def rerank(
        self,
        *,
        candidates: List[dict],
        user_history: List[dict],
        user_context: Optional[str],
        top_k: int,
        enforce_diversity: bool = True,
    ) -> dict:
        if not candidates:
            return {"ranked_items": [], "rerank_summary": "No candidates to rerank."}

        X = []
        tie_break = []
        for c in candidates:
            genre_id = float(
                self.artifacts.genre_to_id.get((c.get("genre") or "unknown").lower(), 0)
            )
            cf = float(c.get("predicted_rating", 3.0)) / 5.0
            pop = float(self.artifacts.popularity.get(c.get("item_id", ""), 0.0))
            fresh = _norm_year(c.get("year"))
            sim = _similarity_to_likes(user_history, c)
            ctx = _context_overlap(user_context, c)
            intent = _context_intent_bonus(user_context, c)
            # model features + extra inference-time features appended
            X.append([cf, pop, fresh, genre_id, sim, ctx, intent])
            # deterministic tie-breaker: prefer intent/context/sim when model ties
            tie_break.append((intent, ctx, sim, cf, pop, fresh))

        X_np = np.asarray(X, dtype=float)

        # The model was trained on 4 features; extend by folding sim/ctx into cf/pop smoothly.
        # This keeps training simple while still using context at inference.
        X4 = X_np[:, :4].copy()
        # boost CF and popularity using sim/context
        boost = 0.20 * X_np[:, 4] + 0.12 * X_np[:, 5] + 0.25 * X_np[:, 6]
        X4[:, 0] = np.clip(X4[:, 0] + boost, 0.0, 1.2)
        X4[:, 1] = np.clip(X4[:, 1] + 0.5 * boost, 0.0, 1.2)

        scores_raw = np.asarray(self.artifacts.model.predict(X4), dtype=float)
        # If the ranker saturates (common on tiny synthetic data), normalize by rank.
        scores_ui = _rank_normalize(scores_raw)

        # Order: primary = model score desc, secondary = intent/context/sim desc (stable).
        idxs = list(range(len(candidates)))
        idxs.sort(
            key=lambda i: (
                float(scores_raw[i]),
                float(tie_break[i][0]),
                float(tie_break[i][1]),
                float(tie_break[i][2]),
                float(tie_break[i][3]),
            ),
            reverse=True,
        )

        ranked = []
        for idx, pos in enumerate(idxs, start=1):
            item = {**candidates[int(pos)]}
            item["rank"] = idx
            # stable 0..1 score for UI (avoid all 1.00)
            item["score"] = float(scores_ui[int(pos)])
            item.setdefault(
                "reason",
                "Ranked by a local learning-to-rank model using CF score + popularity + freshness + preferences.",
            )
            ranked.append(item)

        if enforce_diversity:
            ranked = _enforce_diversity(ranked, max_per_genre=3)
            # Re-number ranks after any re-ordering.
            for i, it in enumerate(ranked, start=1):
                it["rank"] = i

        return {
            "ranked_items": ranked[:top_k],
            "rerank_summary": "Re-ranked by local LightGBM LambdaRank model (no Claude needed).",
        }

