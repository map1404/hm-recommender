# 🛍️ H&M User Taste Profile Recommender

A hybrid recommender system that learns each customer's fashion taste profile
from purchase history and uses an LLM to generate personalised recommendations
with human-readable explanations.

**Course:** Recommender Systems — Final Project
**Team:** Ponnie (PM), Akash (Data), Tianshi (CF), Fuwei (LLM/Content), Lei (UI)

---

## Demo (Quick Start — No Dataset Required)

```bash
# 1. Clone and install
git clone https://github.com/your-org/hm-recommender.git
cd hm-recommender
conda env create -f environment.yml
conda activate hm-recommender

# 2. Generate synthetic demo artifacts (no H&M dataset needed)
python src/demo_artifacts.py

# 3. Launch the app
PYTHONPATH=. streamlit run app/main.py
```

Open http://localhost:8501 and select a demo customer from the sidebar.

---

## Full Pipeline (with H&M Dataset)

### Prerequisites
- `GROQ_API_KEY` in your `.env` file (free at https://console.groq.com)

### Step 1 — Download Dataset (no Kaggle login required)
```bash
# Full dataset
python src/download_data.py

# Or smaller sample for local dev
python src/download_data.py --sample-transactions 500000
```

### Step 2 — Preprocessing
```bash
python src/preprocessing.py
```

### Step 3 — Train ALS Model
```bash
python src/training.py
# Optional flags:
python src/training.py --factors 64 --epochs 20 --regularization 0.01
```

### Step 4 — Build Content Embeddings
```bash
python src/content_based.py
```

### Step 5 — Evaluate (Optional)
```bash
python src/evaluation.py
```
Prints Precision@10, Recall@10, NDCG@10 on a temporal held-out test set.

### Step 6 — Generate Demo Cache
```bash
python src/inference.py --cache
```

### Step 7 — Launch App
```bash
PYTHONPATH=. streamlit run app/main.py

# Live LLM mode (real-time, ~30s latency):
PYTHONPATH=. streamlit run app/main.py -- --live
```

---

## Docker

```bash
docker build -t hm-recommender .
docker run -p 8501:8501 -e GROQ_API_KEY=$GROQ_API_KEY hm-recommender
```

---

## Repository Structure

```
hm-recommender/
├── app/
│   ├── main.py                  # Streamlit entry point
│   └── pages/
│       ├── profile.py           # Customer profile + charts
│       ├── recommendations.py   # Item grid with images + explanations
│       └── explore.py           # Style archetypes browser
├── data/
│   ├── EDA.ipynb                # Exploratory data analysis
│   ├── raw/                     # Raw H&M CSVs
│   └── processed/               # Parquet files (git-ignored)
├── models/
│   ├── als_model.pkl            # Trained ALS model
│   ├── article_embeddings.npy   # Sentence-transformer embeddings
│   └── user_item_matrix.npz     # Sparse user-item matrix
├── demo_cache/
│   ├── taste_profiles.json      # Pre-generated taste profiles
│   ├── explanations.json        # Pre-generated item explanations
│   └── recommendations.json     # Pre-generated recommendations
├── src/
│   ├── download_data.py         # Download H&M CSVs (no Kaggle login)
│   ├── preprocessing.py         # Data cleaning + feature engineering
│   ├── training.py              # ALS model training
│   ├── content_based.py         # Article embeddings + cosine similarity
│   ├── hybrid.py                # Weighted ALS + content scoring
│   ├── llm.py                   # Taste profile + explanation generation
│   ├── inference.py             # End-to-end pipeline + cache generation
│   ├── evaluation.py            # Precision@K, Recall@K, NDCG@K
│   ├── storage.py               # Parquet/pickle read-write helpers
│   └── demo_artifacts.py        # Synthetic demo data generator
├── environment.yml
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## Architecture

```
Purchase History
      │
      ▼
┌─────────────────────────────────────────┐
│             Hybrid Pipeline             │
│                                         │
│  ALS (Collaborative Filtering)  ──┐     │
│  all-MiniLM-L6-v2 (Content)  ────┤  60/40 blend
│                                   ▼     │
│         Combined Score Ranking          │
└──────────────────┬──────────────────────┘
                   │ Top-10 candidates
                   ▼
          ┌─────────────────┐
          │   Groq LLM       │
          │  Taste Profile   │
          │  Per-item Expl.  │
          └────────┬─────────┘
                   ▼
           Streamlit UI
    (Profile | Recs | Explore)
```

---

## Team Contributions

| Member | Role | Key Deliverables |
|---|---|---|
| Ponnie | PM + Report Lead | README, report PDF, evaluation metrics |
| Akash | Data Engineer | EDA.ipynb, preprocessing.py, download_data.py |
| Tianshi | ML Developer A | training.py, evaluation.py, ALS model |
| Fuwei | ML Developer B | content_based.py, hybrid.py, llm.py, inference.py |
| Lei | Frontend + Demo Lead | app/, Dockerfile, presentation slides |

---

## Success Criteria

- **Recognition test:** ≥4/5 testers say "yes, that's me" about their taste profile.
- **Visible coherence:** recommended items visually cluster around the stated style.
- **Grounded explanations:** each LLM explanation references ≥1 concrete attribute.
- **Demo robustness:** runs end-to-end offline on the curated customer set.

---

## Evaluation Results

| Metric | Score |
|---|---|
| Precision@10 | Run `python src/evaluation.py` |
| Recall@10 | Run `python src/evaluation.py` |
| NDCG@10 | Run `python src/evaluation.py` |
