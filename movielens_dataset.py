"""
MovieLens-20M Dataset Loader.

Replaces the synthetic dataset.py with real MovieLens data.

Supports three modes:
  1. MovieLens-20M  (20M ratings, 138K users, 27K movies) — production
  2. MovieLens-1M   (1M ratings,  6K users,  4K movies)  — development
  3. MovieLens-100K (100K ratings, 943 users, 1682 movies) — quick testing
  4. Synthetic fallback (original 10-user data)           — zero-setup demo

Download links:
  20M  → https://grouplens.org/datasets/movielens/20m/
  1M   → https://grouplens.org/datasets/movielens/1m/
  100K → https://grouplens.org/datasets/movielens/100k/

After downloading, unzip to ./data/ml-20m/, ./data/ml-1m/, or ./data/ml-100k/

Usage:
    from movielens_dataset import load_dataset, get_user_ids

    # Auto-detects which dataset is available (20M > 1M > 100K > synthetic)
    ratings, items = load_dataset()

    # Force a specific dataset
    ratings, items = load_dataset("ml-1m")
"""

from __future__ import annotations

import os
import csv
import random
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ---------------------------------------------------------------------------
# Data directory — relative to this file
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.path.dirname(__file__)) / "data"


# ---------------------------------------------------------------------------
# MovieLens-20M loader
# ---------------------------------------------------------------------------

def _load_ml20m(data_dir: Path) -> Tuple[List[dict], List[dict]]:
    """
    Load MovieLens-20M from ./data/ml-20m/
    Files needed: ratings.csv, movies.csv, tags.csv (optional)

    ratings.csv columns : userId, movieId, rating, timestamp
    movies.csv columns  : movieId, title, genres
    """
    ratings_path = data_dir / "ml-20m" / "ratings.csv"
    movies_path  = data_dir / "ml-20m" / "movies.csv"

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            f"MovieLens-20M not found at {data_dir / 'ml-20m'}.\n"
            "Download from https://grouplens.org/datasets/movielens/20m/ "
            "and unzip to ./data/ml-20m/"
        )

    print("Loading MovieLens-20M movies...")
    items = _parse_movies_csv(movies_path)

    print("Loading MovieLens-20M ratings (this may take ~30 seconds)...")
    ratings = _parse_ratings_csv(ratings_path)

    print(f"✓ MovieLens-20M: {len(ratings):,} ratings, {len(items):,} movies")
    return ratings, items


def _load_ml1m(data_dir: Path) -> Tuple[List[dict], List[dict]]:
    """
    Load MovieLens-1M from ./data/ml-1m/
    Files: ratings.dat, movies.dat  (:: separated)
    """
    ratings_path = data_dir / "ml-1m" / "ratings.dat"
    movies_path  = data_dir / "ml-1m" / "movies.dat"

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            f"MovieLens-1M not found at {data_dir / 'ml-1m'}.\n"
            "Download from https://grouplens.org/datasets/movielens/1m/"
        )

    print("Loading MovieLens-1M...")
    items   = _parse_movies_dat(movies_path)
    ratings = _parse_ratings_dat(ratings_path)
    print(f"✓ MovieLens-1M: {len(ratings):,} ratings, {len(items):,} movies")
    return ratings, items


def _load_ml100k(data_dir: Path) -> Tuple[List[dict], List[dict]]:
    """
    Load MovieLens-100K from ./data/ml-100k/
    Files: u.data, u.item
    """
    ratings_path = data_dir / "ml-100k" / "u.data"
    movies_path  = data_dir / "ml-100k" / "u.item"

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            f"MovieLens-100K not found at {data_dir / 'ml-100k'}.\n"
            "Download from https://grouplens.org/datasets/movielens/100k/"
        )

    print("Loading MovieLens-100K...")
    items   = _parse_100k_items(movies_path)
    ratings = _parse_100k_ratings(ratings_path)
    print(f"✓ MovieLens-100K: {len(ratings):,} ratings, {len(items):,} movies")
    return ratings, items


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

_ML_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def _parse_movies_csv(path: Path) -> List[dict]:
    items = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genres_raw = row.get("genres", "")
            genre_list = [g for g in genres_raw.split("|") if g and g != "(no genres listed)"]
            primary_genre = genre_list[0] if genre_list else "Drama"

            title_raw = row.get("title", "Unknown")
            year      = _extract_year(title_raw)
            title     = _clean_title(title_raw)

            items.append({
                "item_id" : f"m{row['movieId']}",
                "title"   : title,
                "genre"   : primary_genre,
                "year"    : year,
                "tags"    : [g.lower().replace("-", "_") for g in genre_list],
            })
    return items


def _parse_ratings_csv(
    path: Path,
    sample_users: Optional[int] = None,
    min_ratings_per_user: int = 10,
) -> List[dict]:
    """
    Parse ratings.csv.
    sample_users: if set, randomly sample this many users (speeds up training).
    """
    # First pass: collect all ratings
    raw: Dict[str, List[dict]] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = f"u{row['userId']}"
            raw.setdefault(uid, []).append({
                "user_id" : uid,
                "item_id" : f"m{row['movieId']}",
                "rating"  : float(row["rating"]),
            })

    # Filter users with too few ratings
    qualified = {uid: rs for uid, rs in raw.items() if len(rs) >= min_ratings_per_user}

    # Optionally sample
    if sample_users and len(qualified) > sample_users:
        rng      = random.Random(42)
        sampled  = rng.sample(list(qualified.keys()), sample_users)
        qualified = {u: qualified[u] for u in sampled}

    ratings = [r for rs in qualified.values() for r in rs]
    return ratings


def _parse_movies_dat(path: Path) -> List[dict]:
    items = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            movie_id, title_raw, genres_raw = parts[0], parts[1], parts[2]
            genre_list    = [g for g in genres_raw.split("|") if g]
            primary_genre = genre_list[0] if genre_list else "Drama"
            year          = _extract_year(title_raw)
            title         = _clean_title(title_raw)
            items.append({
                "item_id" : f"m{movie_id}",
                "title"   : title,
                "genre"   : primary_genre,
                "year"    : year,
                "tags"    : [g.lower().replace("-", "_") for g in genre_list],
            })
    return items


def _parse_ratings_dat(path: Path) -> List[dict]:
    ratings = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            ratings.append({
                "user_id" : f"u{parts[0]}",
                "item_id" : f"m{parts[1]}",
                "rating"  : float(parts[2]),
            })
    return ratings


# 100K genre indices (columns 5-23 in u.item are binary genre flags)
_100K_GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def _parse_100k_items(path: Path) -> List[dict]:
    items = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 24:
                continue
            movie_id  = parts[0]
            title_raw = parts[1]
            year      = _extract_year(title_raw) or (int(parts[2].split("-")[-1]) if parts[2] else 2000)
            title     = _clean_title(title_raw)

            genre_list = [
                _100K_GENRE_COLS[i - 5]
                for i in range(5, min(24, len(parts)))
                if parts[i] == "1"
            ]
            primary_genre = genre_list[0] if genre_list else "Drama"

            items.append({
                "item_id" : f"m{movie_id}",
                "title"   : title,
                "genre"   : primary_genre,
                "year"    : year,
                "tags"    : [g.lower() for g in genre_list],
            })
    return items


def _parse_100k_ratings(path: Path) -> List[dict]:
    ratings = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            ratings.append({
                "user_id" : f"u{parts[0]}",
                "item_id" : f"m{parts[1]}",
                "rating"  : float(parts[2]),
            })
    return ratings


# ---------------------------------------------------------------------------
# Title utilities
# ---------------------------------------------------------------------------

def _extract_year(title: str) -> int:
    """Extract year from 'Movie Title (1994)' format."""
    import re
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        return int(match.group(1))
    return 2000


def _clean_title(title: str) -> str:
    """Remove year suffix: 'Toy Story (1995)' → 'Toy Story'."""
    import re
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


# ---------------------------------------------------------------------------
# Synthetic fallback (original dataset.py data)
# ---------------------------------------------------------------------------

_SYNTHETIC_MOVIES = [
    {"item_id": "m1",  "title": "The Shawshank Redemption", "genre": "Drama",     "year": 1994, "tags": ["prison", "hope", "friendship"]},
    {"item_id": "m2",  "title": "The Godfather",             "genre": "Crime",     "year": 1972, "tags": ["mafia", "family", "classic"]},
    {"item_id": "m3",  "title": "The Dark Knight",           "genre": "Action",    "year": 2008, "tags": ["superhero", "batman", "joker"]},
    {"item_id": "m4",  "title": "Pulp Fiction",              "genre": "Crime",     "year": 1994, "tags": ["nonlinear", "cult", "violence"]},
    {"item_id": "m5",  "title": "Forrest Gump",              "genre": "Drama",     "year": 1994, "tags": ["inspirational", "history", "heartwarming"]},
    {"item_id": "m6",  "title": "Inception",                 "genre": "Sci-Fi",    "year": 2010, "tags": ["dreams", "heist", "mind-bending"]},
    {"item_id": "m7",  "title": "The Matrix",                "genre": "Sci-Fi",    "year": 1999, "tags": ["cyberpunk", "action", "philosophy"]},
    {"item_id": "m8",  "title": "Goodfellas",                "genre": "Crime",     "year": 1990, "tags": ["mafia", "biography", "violence"]},
    {"item_id": "m9",  "title": "Interstellar",              "genre": "Sci-Fi",    "year": 2014, "tags": ["space", "time", "emotional"]},
    {"item_id": "m10", "title": "Fight Club",                "genre": "Drama",     "year": 1999, "tags": ["twist", "anarchy", "identity"]},
    {"item_id": "m11", "title": "The Silence of the Lambs",  "genre": "Thriller",  "year": 1991, "tags": ["serial-killer", "psychological", "FBI"]},
    {"item_id": "m12", "title": "Schindler's List",          "genre": "Drama",     "year": 1993, "tags": ["holocaust", "history", "biography"]},
    {"item_id": "m13", "title": "The Lord of the Rings",     "genre": "Fantasy",   "year": 2001, "tags": ["epic", "adventure", "tolkien"]},
    {"item_id": "m14", "title": "Star Wars: A New Hope",     "genre": "Sci-Fi",    "year": 1977, "tags": ["space", "jedi", "classic"]},
    {"item_id": "m15", "title": "Avengers: Endgame",         "genre": "Action",    "year": 2019, "tags": ["superhero", "MCU", "epic"]},
    {"item_id": "m16", "title": "Parasite",                  "genre": "Thriller",  "year": 2019, "tags": ["class", "korean", "twist"]},
    {"item_id": "m17", "title": "Spirited Away",             "genre": "Animation", "year": 2001, "tags": ["anime", "magical", "miyazaki"]},
    {"item_id": "m18", "title": "The Lion King",             "genre": "Animation", "year": 1994, "tags": ["disney", "family", "classic"]},
    {"item_id": "m19", "title": "Titanic",                   "genre": "Romance",   "year": 1997, "tags": ["love", "disaster", "epic"]},
    {"item_id": "m20", "title": "Toy Story",                 "genre": "Animation", "year": 1995, "tags": ["pixar", "friendship", "family"]},
    {"item_id": "m21", "title": "Se7en",                     "genre": "Thriller",  "year": 1995, "tags": ["detective", "dark", "twist"]},
    {"item_id": "m22", "title": "Back to the Future",        "genre": "Sci-Fi",    "year": 1985, "tags": ["time-travel", "comedy", "classic"]},
    {"item_id": "m23", "title": "Jurassic Park",             "genre": "Adventure", "year": 1993, "tags": ["dinosaurs", "action", "sci-fi"]},
    {"item_id": "m24", "title": "The Truman Show",           "genre": "Drama",     "year": 1998, "tags": ["reality", "philosophical", "unique"]},
    {"item_id": "m25", "title": "Coco",                      "genre": "Animation", "year": 2017, "tags": ["family", "music", "culture"]},
    {"item_id": "m26", "title": "Knives Out",                "genre": "Mystery",   "year": 2019, "tags": ["whodunit", "comedy", "modern"]},
    {"item_id": "m27", "title": "Whiplash",                  "genre": "Drama",     "year": 2014, "tags": ["music", "ambition", "intense"]},
    {"item_id": "m28", "title": "Get Out",                   "genre": "Horror",    "year": 2017, "tags": ["race", "social", "twist"]},
    {"item_id": "m29", "title": "Everything Everywhere",     "genre": "Sci-Fi",    "year": 2022, "tags": ["multiverse", "family", "absurd"]},
    {"item_id": "m30", "title": "Dune",                      "genre": "Sci-Fi",    "year": 2021, "tags": ["epic", "desert", "politics"]},
]

_SYNTHETIC_USER_PROFILES = {
    "alice":    {"preferred_genres": ["Drama", "Sci-Fi"],     "avg_rating": 4.0},
    "bob":      {"preferred_genres": ["Action", "Sci-Fi"],    "avg_rating": 3.5},
    "carol":    {"preferred_genres": ["Animation", "Drama"],  "avg_rating": 4.2},
    "dave":     {"preferred_genres": ["Crime", "Thriller"],   "avg_rating": 3.8},
    "eve":      {"preferred_genres": ["Sci-Fi", "Fantasy"],   "avg_rating": 4.5},
    "frank":    {"preferred_genres": ["Romance", "Drama"],    "avg_rating": 3.2},
    "grace":    {"preferred_genres": ["Horror", "Thriller"],  "avg_rating": 3.9},
    "henry":    {"preferred_genres": ["Mystery", "Crime"],    "avg_rating": 4.1},
    "isabella": {"preferred_genres": ["Animation", "Romance"],"avg_rating": 4.3},
    "jack":     {"preferred_genres": ["Action", "Adventure"], "avg_rating": 3.6},
    "karen":    {"preferred_genres": ["Crime", "Mystery"],    "avg_rating": 4.0},
    "leo":      {"preferred_genres": ["Thriller", "Horror"],  "avg_rating": 3.7},
    "maya":     {"preferred_genres": ["Drama", "Romance"],    "avg_rating": 4.1},
    "noah":     {"preferred_genres": ["Sci-Fi", "Action"],    "avg_rating": 3.9},
    "olivia":   {"preferred_genres": ["Animation", "Fantasy"],"avg_rating": 4.4},
}


def _synthetic_ratings(seed: int = 42, n_per_user: int = 20) -> List[dict]:
    rng     = random.Random(seed)
    ratings = []
    for uid, profile in _SYNTHETIC_USER_PROFILES.items():
        preferred = set(profile["preferred_genres"])
        sampled   = rng.sample(_SYNTHETIC_MOVIES, min(n_per_user, len(_SYNTHETIC_MOVIES)))
        for item in sampled:
            base = profile["avg_rating"]
            if item["genre"] in preferred:
                score = base + rng.uniform(0.3, 1.0)
            else:
                score = base + rng.uniform(-1.5, 0.5)
            rating = round(max(1.0, min(5.0, score)) * 2) / 2
            ratings.append({"user_id": uid, "item_id": item["item_id"], "rating": rating})

    # Cross-user overlap
    for _ in range(80):
        u1, u2 = rng.sample(list(_SYNTHETIC_USER_PROFILES.keys()), 2)
        shared = rng.choice(_SYNTHETIC_MOVIES)
        for uid in [u1, u2]:
            if not any(r["user_id"] == uid and r["item_id"] == shared["item_id"] for r in ratings):
                ratings.append({"user_id": uid, "item_id": shared["item_id"], "rating": 4.0})
    return ratings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    variant: str = "auto",
    sample_users: Optional[int] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Load a MovieLens dataset (or synthetic fallback).

    Args:
        variant     : "auto" | "ml-20m" | "ml-1m" | "ml-100k" | "synthetic"
                      "auto" tries 20M → 1M → 100K → synthetic
        sample_users: For large datasets, randomly sample this many users.
                      Recommended: 20000 for 20M, None for 1M/100K.

    Returns:
        (ratings, items)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if variant == "auto":
        for v in ("ml-20m", "ml-1m", "ml-100k"):
            try:
                return load_dataset(v, sample_users=sample_users)
            except FileNotFoundError:
                continue
        print("No MovieLens data found — using synthetic fallback.")
        print("To use real data, download from https://grouplens.org/datasets/movielens/")
        return load_dataset("synthetic")

    if variant == "ml-20m":
        ratings, items = _load_ml20m(DATA_DIR)
        if sample_users:
            # Re-sample users from the already-parsed ratings
            all_users = list({r["user_id"] for r in ratings})
            rng       = random.Random(42)
            keep      = set(rng.sample(all_users, min(sample_users, len(all_users))))
            ratings   = [r for r in ratings if r["user_id"] in keep]
            print(f"  Sampled {len(keep):,} users → {len(ratings):,} ratings")
        return ratings, items

    if variant == "ml-1m":
        return _load_ml1m(DATA_DIR)

    if variant == "ml-100k":
        return _load_ml100k(DATA_DIR)

    if variant == "synthetic":
        return _synthetic_ratings(), list(_SYNTHETIC_MOVIES)

    raise ValueError(f"Unknown dataset variant: {variant!r}. Use 'auto', 'ml-20m', 'ml-1m', 'ml-100k', or 'synthetic'.")


def get_user_ids(ratings: Optional[List[dict]] = None) -> List[str]:
    """Return sorted list of user IDs."""
    if ratings is not None:
        return sorted({r["user_id"] for r in ratings})
    # Fallback: synthetic user IDs
    return sorted(_SYNTHETIC_USER_PROFILES.keys())
