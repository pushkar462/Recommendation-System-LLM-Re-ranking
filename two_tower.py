"""
Stage 1: Two-Tower Neural Retrieval Model.

Architecture used by YouTube, Google, Pinterest for candidate generation.
Replaces matrix_factorization.py entirely.

Two separate MLP towers:
  - User tower  : encodes user history + profile → user_embedding (64d)
  - Item tower  : encodes item features           → item_embedding (64d)

Relevance score = dot_product(user_embedding, item_embedding)

Training: BPR pairwise loss (positive item should score > negative item)

Works with:
  - Synthetic data (dataset.py)
  - MovieLens-20M (movielens_dataset.py)
"""

from __future__ import annotations

import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_DIM      = 64      # shared embedding dimension for both towers
USER_HIDDEN    = [128, 64]
ITEM_HIDDEN    = [128, 64]
DROPOUT        = 0.2
LR             = 1e-3
EPOCHS         = 30
BATCH_SIZE     = 512
NEG_SAMPLES    = 4       # negatives per positive in training


# ---------------------------------------------------------------------------
# Towers
# ---------------------------------------------------------------------------

class UserTower(nn.Module):
    """
    Encodes a user as a dense vector.

    Input features (all normalised to 0..1):
        user_id_onehot  : one-hot or learned embedding (handled externally as lookup)
        avg_rating      : mean of user's ratings
        n_ratings       : number of items rated (log-normalised)
        genre_prefs     : distribution over N genres (soft histogram)
    """

    def __init__(self, n_users: int, n_genres: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.n_genres  = n_genres
        self.embed_dim = embed_dim

        # Learned user ID embedding
        self.user_emb = nn.Embedding(n_users + 1, 32, padding_idx=0)

        # Dense layers on top of concat(user_emb, stat_features, genre_prefs)
        in_dim = 32 + 2 + n_genres
        layers = []
        prev = in_dim
        for h in USER_HIDDEN:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = h
        layers.append(nn.Linear(prev, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        user_idx: torch.Tensor,          # (B,)
        avg_rating: torch.Tensor,        # (B,)
        n_ratings_norm: torch.Tensor,    # (B,)
        genre_prefs: torch.Tensor,       # (B, n_genres)
    ) -> torch.Tensor:                   # (B, embed_dim)
        uid = self.user_emb(user_idx)    # (B, 32)
        stats = torch.stack([avg_rating, n_ratings_norm], dim=1)  # (B, 2)
        x = torch.cat([uid, stats, genre_prefs], dim=1)
        return F.normalize(self.net(x), dim=-1)


class ItemTower(nn.Module):
    """
    Encodes a movie/item as a dense vector.

    Input features:
        item_id_onehot  : learned item embedding
        genre_id        : genre as integer index
        year_norm       : release year normalised to 0..1
        popularity_norm : log-normalised rating count
        avg_item_rating : mean rating for this item
    """

    def __init__(self, n_items: int, n_genres: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, 32, padding_idx=0)

        # genre one-hot
        in_dim = 32 + n_genres + 3   # item_emb + genre_onehot + year + pop + avg_rating
        layers = []
        prev = in_dim
        for h in ITEM_HIDDEN:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = h
        layers.append(nn.Linear(prev, embed_dim))
        self.net = nn.Sequential(*layers)
        self.n_genres = n_genres

    def forward(
        self,
        item_idx: torch.Tensor,          # (B,)
        genre_idx: torch.Tensor,         # (B,)  integer
        year_norm: torch.Tensor,         # (B,)
        pop_norm: torch.Tensor,          # (B,)
        avg_item_rating: torch.Tensor,   # (B,)
    ) -> torch.Tensor:                   # (B, embed_dim)
        iid  = self.item_emb(item_idx)
        goh  = F.one_hot(genre_idx, self.n_genres).float()
        rest = torch.stack([year_norm, pop_norm, avg_item_rating], dim=1)
        x    = torch.cat([iid, goh, rest], dim=1)
        return F.normalize(self.net(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_genres: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.user_tower = UserTower(n_users, n_genres, embed_dim)
        self.item_tower = ItemTower(n_items, n_genres, embed_dim)
        self.embed_dim  = embed_dim

    def score(
        self,
        user_feats: dict,
        item_feats: dict,
    ) -> torch.Tensor:
        u_emb = self.user_tower(**user_feats)
        i_emb = self.item_tower(**item_feats)
        return (u_emb * i_emb).sum(dim=-1)   # dot product

    def forward(self, user_feats, pos_item_feats, neg_item_feats):
        u_emb   = self.user_tower(**user_feats)
        pos_emb = self.item_tower(**pos_item_feats)
        neg_emb = self.item_tower(**neg_item_feats)
        pos_score = (u_emb * pos_emb).sum(dim=-1)
        neg_score = (u_emb * neg_emb).sum(dim=-1)
        return pos_score, neg_score


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class PairwiseDataset(Dataset):
    def __init__(
        self,
        ratings: List[dict],
        user_feature_map: Dict[str, dict],
        item_feature_map: Dict[str, dict],
        neg_samples: int = NEG_SAMPLES,
        seed: int = 42,
    ):
        self.rng         = random.Random(seed)
        self.item_ids    = list(item_feature_map.keys())
        self.user_feats  = user_feature_map
        self.item_feats  = item_feature_map
        self.neg_samples = neg_samples

        # Index positive interactions per user
        self.pairs: List[Tuple[str, str]] = []
        seen = set()
        by_user: Dict[str, List[str]] = {}
        for r in ratings:
            if float(r.get("rating", 0)) >= 4.0:
                uid, iid = r["user_id"], r["item_id"]
                if (uid, iid) not in seen and iid in item_feature_map:
                    seen.add((uid, iid))
                    self.pairs.append((uid, iid))
                    by_user.setdefault(uid, []).append(iid)

        self.by_user = by_user

    def __len__(self):
        return len(self.pairs) * self.neg_samples

    def __getitem__(self, idx):
        pair_idx = idx // self.neg_samples
        uid, pos_iid = self.pairs[pair_idx]

        # Sample a negative item (not in user's positives)
        seen_pos = set(self.by_user.get(uid, []))
        neg_iid  = self.rng.choice(self.item_ids)
        for _ in range(10):
            if neg_iid not in seen_pos:
                break
            neg_iid = self.rng.choice(self.item_ids)

        return (
            self.user_feats[uid],
            self.item_feats[pos_iid],
            self.item_feats[neg_iid],
        )


def _collate(batch):
    """Collate list of (user_feat, pos_feat, neg_feat) dicts into batched tensors."""
    def stack(dicts, key):
        vals = [d[key] for d in dicts]
        t    = vals[0]
        if isinstance(t, torch.Tensor):
            return torch.stack(vals)
        return torch.tensor(vals)

    user_keys     = list(batch[0][0].keys())
    pos_item_keys = list(batch[0][1].keys())

    user_batch     = {k: stack([b[0] for b in batch], k) for k in user_keys}
    pos_item_batch = {k: stack([b[1] for b in batch], k) for k in pos_item_keys}
    neg_item_batch = {k: stack([b[2] for b in batch], k) for k in pos_item_keys}

    return user_batch, pos_item_batch, neg_item_batch


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _norm_year(year, lo=1970, hi=2026):
    try:
        y = max(lo, min(hi, int(year)))
    except Exception:
        y = 2000
    return (y - lo) / (hi - lo)


def build_user_feature_map(
    ratings: List[dict],
    items_by_id: Dict[str, dict],
    user_id_map: Dict[str, int],
    genre_to_id: Dict[str, int],
) -> Dict[str, dict]:
    n_genres = len(genre_to_id)
    by_user: Dict[str, List[dict]] = {}
    for r in ratings:
        by_user.setdefault(r["user_id"], []).append(r)

    feats = {}
    for uid, rs in by_user.items():
        if uid not in user_id_map:
            continue
        ratings_vals = [float(r["rating"]) for r in rs if r.get("rating")]
        avg_r  = float(np.mean(ratings_vals)) / 5.0 if ratings_vals else 0.5
        n_r    = float(np.log1p(len(ratings_vals))) / 10.0

        genre_counts = np.zeros(n_genres, dtype=np.float32)
        for r in rs:
            meta  = items_by_id.get(r["item_id"], {})
            gname = (meta.get("genre") or "unknown").lower()
            gid   = genre_to_id.get(gname, 0)
            genre_counts[gid] += float(r.get("rating", 3.0))
        total = genre_counts.sum()
        genre_prefs = genre_counts / (total + 1e-8)

        feats[uid] = {
            "user_idx"       : torch.tensor(user_id_map[uid] + 1, dtype=torch.long),
            "avg_rating"     : torch.tensor(avg_r,  dtype=torch.float32),
            "n_ratings_norm" : torch.tensor(n_r,    dtype=torch.float32),
            "genre_prefs"    : torch.tensor(genre_prefs, dtype=torch.float32),
        }
    return feats


def build_item_feature_map(
    items: List[dict],
    ratings: List[dict],
    item_id_map: Dict[str, int],
    genre_to_id: Dict[str, int],
) -> Dict[str, dict]:
    n_users = len({r["user_id"] for r in ratings}) or 1
    pop_counts: Dict[str, int] = {}
    sum_ratings: Dict[str, float] = {}
    cnt_ratings: Dict[str, int] = {}
    for r in ratings:
        iid = r["item_id"]
        pop_counts[iid]   = pop_counts.get(iid, 0) + 1
        sum_ratings[iid]  = sum_ratings.get(iid, 0.0) + float(r.get("rating", 3.0))
        cnt_ratings[iid]  = cnt_ratings.get(iid, 0) + 1

    feats = {}
    for it in items:
        iid = it["item_id"]
        if iid not in item_id_map:
            continue
        gname  = (it.get("genre") or "unknown").lower()
        gid    = genre_to_id.get(gname, 0)
        year   = _norm_year(it.get("year", 2000))
        pop    = float(np.log1p(pop_counts.get(iid, 0))) / float(np.log1p(n_users))
        avg_ir = (sum_ratings.get(iid, 3.0) / max(1, cnt_ratings.get(iid, 1))) / 5.0

        feats[iid] = {
            "item_idx"       : torch.tensor(item_id_map[iid] + 1, dtype=torch.long),
            "genre_idx"      : torch.tensor(gid,    dtype=torch.long),
            "year_norm"      : torch.tensor(year,   dtype=torch.float32),
            "pop_norm"       : torch.tensor(pop,    dtype=torch.float32),
            "avg_item_rating": torch.tensor(avg_ir, dtype=torch.float32),
        }
    return feats


# ---------------------------------------------------------------------------
# Trainer & index builder
# ---------------------------------------------------------------------------

def bpr_loss(pos_scores, neg_scores):
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def train_two_tower(
    ratings: List[dict],
    items: List[dict],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    embed_dim: int = EMBED_DIM,
    seed: int = 42,
    device: str = "cpu",
) -> "TwoTowerRetriever":
    torch.manual_seed(seed)
    random.seed(seed)

    # Build index maps
    user_ids  = sorted({r["user_id"] for r in ratings})
    item_ids  = sorted({it["item_id"] for it in items})
    genres    = sorted({(it.get("genre") or "unknown").lower() for it in items})

    user_id_map  = {uid: i for i, uid in enumerate(user_ids)}
    item_id_map  = {iid: i for i, iid in enumerate(item_ids)}
    genre_to_id  = {g: i for i, g in enumerate(genres)}
    items_by_id  = {it["item_id"]: it for it in items}

    print(f"Building feature maps for {len(user_ids)} users, {len(item_ids)} items, {len(genres)} genres...")
    user_feats = build_user_feature_map(ratings, items_by_id, user_id_map, genre_to_id)
    item_feats = build_item_feature_map(items, ratings, item_id_map, genre_to_id)

    # Dataset & loader
    ds     = PairwiseDataset(ratings, user_feats, item_feats, neg_samples=NEG_SAMPLES, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate, num_workers=0)

    # Model
    model = TwoTowerModel(
        n_users=len(user_ids),
        n_items=len(item_ids),
        n_genres=len(genres),
        embed_dim=embed_dim,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Training two-tower model for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for user_batch, pos_batch, neg_batch in loader:
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            pos_batch  = {k: v.to(device) for k, v in pos_batch.items()}
            neg_batch  = {k: v.to(device) for k, v in neg_batch.items()}

            pos_s, neg_s = model(user_batch, pos_batch, neg_batch)
            loss = bpr_loss(pos_s, neg_s)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss: {total_loss / max(1, len(loader)):.4f}")

    model.eval()

    retriever = TwoTowerRetriever(
        model        = model,
        user_id_map  = user_id_map,
        item_id_map  = item_id_map,
        genre_to_id  = genre_to_id,
        user_feats   = user_feats,
        item_feats   = item_feats,
        items_by_id  = items_by_id,
        ratings      = ratings,
        device       = device,
    )
    retriever._build_item_index()
    print("✓ Two-tower model ready.")
    return retriever


# ---------------------------------------------------------------------------
# Retriever (replaces MatrixFactorizationRecommender interface)
# ---------------------------------------------------------------------------

class TwoTowerRetriever:
    """
    Drop-in replacement for MatrixFactorizationRecommender.
    Same public API: get_candidates(user_id, top_k) and get_user_history(user_id).
    """

    def __init__(
        self,
        model: TwoTowerModel,
        user_id_map: Dict[str, int],
        item_id_map: Dict[str, int],
        genre_to_id: Dict[str, int],
        user_feats: Dict[str, dict],
        item_feats: Dict[str, dict],
        items_by_id: Dict[str, dict],
        ratings: List[dict],
        device: str = "cpu",
    ):
        self.model       = model
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.genre_to_id = genre_to_id
        self.user_feats  = user_feats
        self.item_feats  = item_feats
        self.items_by_id = items_by_id
        self.device      = device

        # Rating matrix for history lookup (user_idx → {item_id: rating})
        self._user_ratings: Dict[str, Dict[str, float]] = {}
        for r in ratings:
            self._user_ratings.setdefault(r["user_id"], {})[r["item_id"]] = float(r["rating"])

        self._item_embeddings: Optional[torch.Tensor] = None
        self._item_id_list: List[str] = []

    def _build_item_index(self):
        """Pre-compute all item embeddings for fast ANN retrieval."""
        print("  Building item embedding index...")
        self.model.eval()
        all_ids   = list(self.item_feats.keys())
        all_feats = [self.item_feats[iid] for iid in all_ids]

        batch_size = 512
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_feats), batch_size):
                batch = all_feats[i : i + batch_size]
                batched = {}
                for key in batch[0]:
                    vals = [b[key] for b in batch]
                    batched[key] = torch.stack(vals).to(self.device)
                emb = self.model.item_tower(**batched)
                embeddings.append(emb.cpu())

        self._item_embeddings = torch.cat(embeddings, dim=0)   # (n_items, embed_dim)
        self._item_id_list    = all_ids
        print(f"  Item index built: {len(all_ids)} items × {self._item_embeddings.shape[1]}d")

    def get_candidates(
        self,
        user_id: str,
        top_k: int = 20,
        exclude_seen: bool = True,
    ) -> List[dict]:
        if user_id not in self.user_feats:
            return self._cold_start_candidates(top_k)

        self.model.eval()
        with torch.no_grad():
            ufeats = {k: v.unsqueeze(0).to(self.device) for k, v in self.user_feats[user_id].items()}
            u_emb  = self.model.user_tower(**ufeats).squeeze(0).cpu()  # (embed_dim,)

        scores = (self._item_embeddings @ u_emb).numpy()  # (n_items,)

        # Mask seen items
        if exclude_seen:
            seen = set(self._user_ratings.get(user_id, {}).keys())
            for j, iid in enumerate(self._item_id_list):
                if iid in seen:
                    scores[j] = -1e9

        top_idx    = np.argsort(-scores)[:top_k]
        candidates = []
        for idx in top_idx:
            iid  = self._item_id_list[idx]
            meta = self.items_by_id.get(iid, {})
            candidates.append({
                "item_id"         : iid,
                "predicted_rating": float(np.clip(scores[idx] * 5.0, 1.0, 5.0)),
                **meta,
            })
        return candidates

    def get_user_history(self, user_id: str) -> List[dict]:
        rated = self._user_ratings.get(user_id, {})
        history = []
        for iid, rating in rated.items():
            meta = self.items_by_id.get(iid, {})
            history.append({"item_id": iid, "rating": rating, **meta})
        return sorted(history, key=lambda x: -x["rating"])

    def _cold_start_candidates(self, top_k: int) -> List[dict]:
        """Popularity-based fallback for unseen users."""
        pop: Dict[str, int] = {}
        for rated in self._user_ratings.values():
            for iid in rated:
                pop[iid] = pop.get(iid, 0) + 1
        sorted_items = sorted(pop.keys(), key=lambda i: -pop[i])[:top_k]
        return [
            {"item_id": iid, "predicted_rating": 3.5, **self.items_by_id.get(iid, {})}
            for iid in sorted_items
        ]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state"  : self.model.state_dict(),
            "user_id_map"  : self.user_id_map,
            "item_id_map"  : self.item_id_map,
            "genre_to_id"  : self.genre_to_id,
            "user_feats"   : self.user_feats,
            "item_feats"   : self.item_feats,
            "items_by_id"  : self.items_by_id,
            "item_id_list" : self._item_id_list,
            "item_embeddings": self._item_embeddings,
            "n_users"      : len(self.user_id_map),
            "n_items"       : len(self.item_id_map),
            "n_genres"     : len(self.genre_to_id),
            "embed_dim"    : self.model.embed_dim,
        }, path)
        print(f"✓ Two-tower model saved to {path}")

    @classmethod
    def load(cls, path: str, ratings: List[dict], device: str = "cpu") -> "TwoTowerRetriever":
        data = torch.load(path, map_location=device)
        model = TwoTowerModel(
            n_users   = data["n_users"],
            n_items   = data["n_items"],
            n_genres  = data["n_genres"],
            embed_dim = data["embed_dim"],
        )
        model.load_state_dict(data["model_state"])
        model.eval()
        retriever = cls(
            model       = model,
            user_id_map = data["user_id_map"],
            item_id_map = data["item_id_map"],
            genre_to_id = data["genre_to_id"],
            user_feats  = data["user_feats"],
            item_feats  = data["item_feats"],
            items_by_id = data["items_by_id"],
            ratings     = ratings,
            device      = device,
        )
        retriever._item_embeddings = data["item_embeddings"]
        retriever._item_id_list    = data["item_id_list"]
        print(f"✓ Two-tower model loaded from {path}")
        return retriever
