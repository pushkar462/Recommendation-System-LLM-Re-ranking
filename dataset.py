"""
Synthetic dataset generator.
Produces realistic MovieLens-style or Amazon-style data for demo purposes.
Replace with real data by swapping load_dataset().
"""

import random
from typing import Tuple, List

# ── Movie data ────────────────────────────────────────────────────────────────

MOVIES = [
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

# ── Amazon-style products ─────────────────────────────────────────────────────

PRODUCTS = [
    {"item_id": "p1",  "title": "Sony WH-1000XM5 Headphones",    "genre": "Electronics", "year": 2022, "tags": ["noise-cancelling", "wireless", "premium"]},
    {"item_id": "p2",  "title": "Kindle Paperwhite",              "genre": "Electronics", "year": 2021, "tags": ["e-reader", "portable", "books"]},
    {"item_id": "p3",  "title": "Instant Pot Duo 7-in-1",        "genre": "Kitchen",     "year": 2020, "tags": ["pressure-cooker", "multi-use", "time-saving"]},
    {"item_id": "p4",  "title": "LEGO Architecture Set",          "genre": "Toys",        "year": 2023, "tags": ["creative", "display", "adult-fans"]},
    {"item_id": "p5",  "title": "Anker PowerCore 26800",          "genre": "Electronics", "year": 2022, "tags": ["battery", "portable", "charging"]},
    {"item_id": "p6",  "title": "Yoga Mat Premium Cork",          "genre": "Sports",      "year": 2021, "tags": ["fitness", "eco", "non-slip"]},
    {"item_id": "p7",  "title": "Moleskine Classic Notebook",     "genre": "Stationery",  "year": 2023, "tags": ["journaling", "premium", "writing"]},
    {"item_id": "p8",  "title": "AeroPress Coffee Maker",         "genre": "Kitchen",     "year": 2020, "tags": ["coffee", "portable", "smooth"]},
    {"item_id": "p9",  "title": "Garmin Forerunner 255",          "genre": "Sports",      "year": 2022, "tags": ["GPS", "running", "health"]},
    {"item_id": "p10", "title": "Raspberry Pi 4 Model B",         "genre": "Electronics", "year": 2022, "tags": ["maker", "programming", "DIY"]},
]


# ── Synthetic user profiles ───────────────────────────────────────────────────

USER_PROFILES = {
    "alice":   {"preferred_genres": ["Drama", "Sci-Fi"],    "avg_rating": 4.0},
    "bob":     {"preferred_genres": ["Action", "Sci-Fi"],   "avg_rating": 3.5},
    "carol":   {"preferred_genres": ["Animation", "Drama"], "avg_rating": 4.2},
    "dave":    {"preferred_genres": ["Crime", "Thriller"],  "avg_rating": 3.8},
    "eve":     {"preferred_genres": ["Sci-Fi", "Fantasy"],  "avg_rating": 4.5},
    "frank":   {"preferred_genres": ["Romance", "Drama"],   "avg_rating": 3.2},
    "grace":   {"preferred_genres": ["Horror", "Thriller"], "avg_rating": 3.9},
    "henry":   {"preferred_genres": ["Mystery", "Crime"],   "avg_rating": 4.1},
    "isabella":{"preferred_genres": ["Animation", "Romance"],"avg_rating": 4.3},
    "jack":    {"preferred_genres": ["Action", "Adventure"],"avg_rating": 3.6},
}


def _simulate_ratings(
    items: List[dict],
    seed: int = 42,
    n_ratings_per_user: int = 12,
) -> List[dict]:
    """Generate realistic user-item ratings with genre affinity."""
    rng = random.Random(seed)
    ratings = []

    for user_id, profile in USER_PROFILES.items():
        preferred = set(profile["preferred_genres"])
        sample_items = rng.sample(items, min(n_ratings_per_user, len(items)))

        for item in sample_items:
            base = profile["avg_rating"]
            if item["genre"] in preferred:
                score = base + rng.uniform(0.3, 1.0)
            else:
                score = base + rng.uniform(-1.5, 0.5)
            rating = round(max(1.0, min(5.0, score)) * 2) / 2  # half-star
            ratings.append({
                "user_id": user_id,
                "item_id": item["item_id"],
                "rating": rating,
            })

    # Add cross-user overlap for collaborative signal
    for _ in range(30):
        u1, u2 = rng.sample(list(USER_PROFILES.keys()), 2)
        shared = rng.choice(items)
        for uid in [u1, u2]:
            if not any(r["user_id"] == uid and r["item_id"] == shared["item_id"] for r in ratings):
                ratings.append({"user_id": uid, "item_id": shared["item_id"], "rating": 4.0})

    return ratings


def load_dataset(dataset: str = "movies") -> Tuple[List[dict], List[dict]]:
    """
    Load a dataset.
    
    Args:
        dataset: "movies" (MovieLens-style) or "products" (Amazon-style)
    
    Returns:
        (ratings, item_metadata)
    """
    if dataset == "products":
        items = PRODUCTS
    else:
        items = MOVIES

    ratings = _simulate_ratings(items)
    return ratings, items


def get_user_ids() -> List[str]:
    return list(USER_PROFILES.keys())
