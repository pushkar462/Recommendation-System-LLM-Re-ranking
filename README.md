# Two-Stage Recommendation System + LLM Re-ranking

The same architecture used by **Google Ads, YouTube, and LinkedIn**:
1. **Stage 1 — Collaborative Filtering (Matrix Factorisation)**: Fast, scalable candidate generation
2. **Stage 2 — LLM Re-ranking**: Context-aware re-prioritisation using Claude

---

## Project Structure

```
recsys/
├── data/
│   └── dataset.py          # Synthetic MovieLens/Amazon data + user profiles
├── models/
│   ├── matrix_factorization.py   # SVD-based CF model (Stage 1)
│   └── llm_reranker.py           # Claude API re-ranker (Stage 2)
├── api/
│   └── server.py           # FastAPI REST server
├── demo.html               # Interactive standalone demo (no server needed)
└── requirements.txt
```

---

## Quick Start

### Option A: Interactive Demo (no setup required)
Open `demo.html` in your browser. It runs the full pipeline client-side.

### Option B: Full Python Backend

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
cd recsys
python -m api.server
```

Then visit http://localhost:8000/docs for the interactive API docs.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/users` | List all users |
| GET | `/history/{user_id}` | User rating history |
| GET | `/candidates/{user_id}` | Stage 1: MF candidates only |
| GET | `/recommend/{user_id}` | **Full pipeline** (Stage 1 + 2) |
| POST | `/rerank` | Stage 2 only (bring your own candidates) |

### Example: Full Pipeline

```bash
curl "http://localhost:8000/recommend/alice?user_context=I+want+something+mind-bending&dataset=movies"
```

---

## How It Works

### Stage 1 — Matrix Factorisation (SVD)

The user-item rating matrix **R** is decomposed:

```
R ≈ U · Σ · Vt
```

- **U** (n_users × k): User latent factor matrix
- **Σ** (k × k): Diagonal singular value matrix  
- **Vt** (k × n_items): Item latent factor matrix

For each user, we compute scores for all unseen items:
```
score(u, i) = U[u] · V[i]
```
Top-20 scored items become the candidate set.

**Why SVD?** It captures latent taste signals — users who liked similar movies cluster together in latent space, enabling collaborative signal without explicit features.

### Stage 2 — LLM Re-ranking

Claude receives:
- The 20 MF candidates with their CF scores
- The user's rating history (for taste profile)
- Optional free-text context ("I want something light tonight")

Claude re-orders the candidates considering:
- Alignment with the user's taste profile
- The stated mood/context
- Genre diversity
- Novelty vs familiarity balance

**Why two stages?** Stage 1 is fast (milliseconds) and handles scale (millions of items). Stage 2 is slower and expensive — but it only processes ~20 pre-filtered candidates, making it practical.

---

## Using Real Data

### MovieLens

```python
# Download: https://grouplens.org/datasets/movielens/
import pandas as pd

ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
movies_df = pd.read_csv('ml-latest-small/movies.csv')

ratings = ratings_df.rename(columns={'userId':'user_id','movieId':'item_id'}).to_dict('records')
items = movies_df.rename(columns={'movieId':'item_id'}).assign(
    genre=lambda d: d['genres'].str.split('|').str[0]
).to_dict('records')
```

### Amazon Reviews

```python
# Download: https://nijianmo.github.io/amazon/
import json, gzip

ratings, items = [], {}
with gzip.open('reviews_Electronics_5.json.gz') as f:
    for line in f:
        r = json.loads(line)
        ratings.append({'user_id': r['reviewerID'], 'item_id': r['asin'], 'rating': r['overall']})
```

---

## Key Concepts

| Concept | Where Used | Why It Matters |
|---------|-----------|----------------|
| Matrix Factorisation | Stage 1 | Captures latent user-item affinity |
| SVD / Truncated SVD | Stage 1 | Efficient approximation for sparse matrices |
| Latent Factors | Stage 1 | Hidden taste dimensions (e.g., "prefers dark films") |
| Collaborative Filtering | Stage 1 | "Users like you also liked…" |
| LLM Re-ranking | Stage 2 | Injects semantic understanding + real-time context |
| Embeddings | Both stages | Dense vector representations of users/items |
| Two-stage retrieval | Architecture | Speed at scale + quality at the top |

---

## Industry References

- **Google Ads**: Two-tower retrieval → cross-attention re-ranking
- **YouTube**: Candidate generation (CF) → Deep neural re-ranking
- **LinkedIn**: Graph-based retrieval → contextual re-ranking
- **Netflix**: Matrix factorisation → ensemble re-ranking

These systems all share the same core insight: *cheap retrieval at scale, expensive ranking at the top.*
