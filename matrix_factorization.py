"""
Stage 1: Matrix Factorisation using SVD for collaborative filtering.
Generates top-K candidate items for a given user.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from typing import List, Tuple, Dict
import json


class MatrixFactorizationRecommender:
    """
    Collaborative filtering via truncated SVD (equivalent to latent factor model).
    
    R ≈ U · Σ · Vt
    
    - R: user-item rating matrix (sparse)
    - U: user latent factors
    - Σ: singular values
    - Vt: item latent factors
    """

    def __init__(self, n_factors: int = 20):
        self.n_factors = n_factors
        self.user_factors = None      # shape: (n_users, n_factors)
        self.item_factors = None      # shape: (n_items, n_factors)
        self.sigma = None
        self.user_id_map: Dict[str, int] = {}
        self.item_id_map: Dict[str, int] = {}
        self.reverse_item_map: Dict[int, str] = {}
        self.item_metadata: Dict[str, dict] = {}
        self.rating_matrix: np.ndarray = None
        self.global_mean: float = 0.0

    def fit(self, ratings: List[dict], item_metadata: List[dict]):
        """
        Train the model.
        
        Args:
            ratings: list of {"user_id": ..., "item_id": ..., "rating": ...}
            item_metadata: list of {"item_id": ..., "title": ..., "genre": ..., ...}
        """
        # Index users and items
        user_ids = sorted(set(r["user_id"] for r in ratings))
        item_ids = sorted(set(r["item_id"] for r in ratings))

        self.user_id_map = {uid: i for i, uid in enumerate(user_ids)}
        self.item_id_map = {iid: i for i, iid in enumerate(item_ids)}
        self.reverse_item_map = {i: iid for iid, i in self.item_id_map.items()}

        # Store metadata
        self.item_metadata = {m["item_id"]: m for m in item_metadata}

        n_users = len(user_ids)
        n_items = len(item_ids)

        # Build dense rating matrix (mean-centered)
        R = np.zeros((n_users, n_items))
        for r in ratings:
            u = self.user_id_map[r["user_id"]]
            i = self.item_id_map[r["item_id"]]
            R[u, i] = r["rating"]

        self.global_mean = R[R > 0].mean()

        # Mean-center only observed entries
        R_centered = R.copy()
        R_centered[R > 0] -= self.global_mean

        # SVD on sparse matrix
        R_sparse = csr_matrix(R_centered)
        k = min(self.n_factors, min(n_users, n_items) - 1)
        U, sigma, Vt = svds(R_sparse.astype(float), k=k)

        # Sort by descending singular value
        idx = np.argsort(-sigma)
        U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]

        self.user_factors = U * sigma          # absorb sigma into user factors
        self.item_factors = Vt.T               # shape: (n_items, n_factors)
        self.sigma = sigma
        self.rating_matrix = R

        print(f"✓ MF model trained: {n_users} users × {n_items} items, {k} latent factors")

    def get_candidates(self, user_id: str, top_k: int = 20, exclude_seen: bool = True) -> List[dict]:
        """
        Generate top-K candidate items for a user via dot-product scoring.
        
        Returns list of {"item_id", "title", "genre", "predicted_rating", "metadata"}
        """
        if user_id not in self.user_id_map:
            # Cold-start: return globally popular items
            return self._cold_start_candidates(top_k)

        u_idx = self.user_id_map[user_id]
        user_vec = self.user_factors[u_idx]           # (n_factors,)

        # Score all items
        scores = self.item_factors @ user_vec         # (n_items,)
        scores += self.global_mean                     # re-add mean

        # Exclude already-seen items
        if exclude_seen:
            seen_items = np.where(self.rating_matrix[u_idx] > 0)[0]
            scores[seen_items] = -np.inf

        top_indices = np.argsort(-scores)[:top_k]

        candidates = []
        for idx in top_indices:
            item_id = self.reverse_item_map[idx]
            meta = self.item_metadata.get(item_id, {})
            candidates.append({
                "item_id": item_id,
                "predicted_rating": float(np.clip(scores[idx], 1, 5)),
                **meta,
            })

        return candidates

    def get_user_history(self, user_id: str) -> List[dict]:
        """Return items the user has already rated (for context)."""
        if user_id not in self.user_id_map:
            return []
        u_idx = self.user_id_map[user_id]
        rated = np.where(self.rating_matrix[u_idx] > 0)[0]
        history = []
        for idx in rated:
            item_id = self.reverse_item_map[idx]
            meta = self.item_metadata.get(item_id, {})
            rating = self.rating_matrix[u_idx, idx]
            history.append({"item_id": item_id, "rating": float(rating), **meta})
        return sorted(history, key=lambda x: -x["rating"])

    def _cold_start_candidates(self, top_k: int) -> List[dict]:
        """Fallback: return globally most-rated items."""
        item_counts = (self.rating_matrix > 0).sum(axis=0)
        item_means = np.where(
            item_counts > 0,
            self.rating_matrix.sum(axis=0) / np.maximum(item_counts, 1),
            0,
        )
        top_indices = np.argsort(-item_means)[:top_k]
        candidates = []
        for idx in top_indices:
            item_id = self.reverse_item_map[idx]
            meta = self.item_metadata.get(item_id, {})
            candidates.append({
                "item_id": item_id,
                "predicted_rating": float(item_means[idx]),
                **meta,
            })
        return candidates
