# H&M User Taste Profile Recommender

A hybrid recommendation system that learns each customer's taste profile from purchase history and uses an LLM to generate personalized recommendations with human-readable explanations.

## Project Overview

- **Dataset**: H&M Personalized Fashion Recommendations via a public Hugging Face mirror
- **Model**: Hybrid — ALS collaborative filtering + sentence-transformer content-based filtering
- **LLM**: Taste profile generation + per-item explanation via OpenAI Responses API
- **UI**: Streamlit app with pre-cached demo outputs

## Current data behavior

- The downloader fetches only the tabular raw files: `articles.csv`, `customers.csv`, and `transactions_train.csv`.
- Product images are **not** downloaded locally.
- When the app renders recommendations, it derives the product image URL from the recommended `article_id` and loads that URL directly.
- This keeps setup lighter while still allowing recommended items to display images when the remote image URL is valid.

## Legacy code

- `legacy/scripts/` contains a previous version of the application kept for reference.
- The active implementation in this repository lives in `app/` and `src/`.

## Quickstart

```bash
# 1. Create the conda environment
conda env create -f environment.yml
conda activate hm-recommender

# 2. Download the dataset from the public mirror (no Kaggle login required)
# CSVs only; no images are downloaded
python src/download_data.py

# For a smaller local-dev dataset instead of the full download:
python src/download_data.py --sample-transactions 500000

# 3. Run preprocessing
python src/preprocessing.py

# 4. Train the ALS model
python src/training.py

# 5. Build content embeddings
python src/content_based.py

# 6. Pre-generate demo cache
python src/inference.py --cache

# 7. Launch the Streamlit app
PYTHONPATH=. streamlit run app/main.py
```

## Live inference (optional)

```bash
PYTHONPATH=. streamlit run app/main.py -- --live
```

> Note: live mode adds ~30s latency per query due to real-time LLM calls.

## OpenAI model choice

- Default: `gpt-5-mini` for lower cost than the full GPT-5 model.
- Cheapest option: `gpt-5-nano` if cost matters more than output quality.
- Set the model in `.env` with `OPENAI_MODEL=...`.

## Docker

```bash
docker build -t hm-recommender .
docker run -p 8501:8501 hm-recommender
```

## Render deployment

This repo is configured to deploy on Render as a Docker web service.

- At container startup, it runs `python -m src.render_bootstrap` to:
  - download a sampled real dataset from the public mirror
  - preprocess it
  - train a small ALS model
  - build article embeddings
  - pre-generate cache entries for a small set of real customer IDs
- It then starts Streamlit on `0.0.0.0:$PORT`, which matches Render's web
  service requirements.
- The deployed site does not need local image downloads. Recommendation images
  are loaded by URL at render time.

### Deploy from the repo

1. Push the repo to GitHub.
2. In Render, create a new Blueprint or Web Service from this repository.
3. Render will detect [render.yaml](/Users/ponnimuthukumarasamy/Desktop/hm-recommender/render.yaml).
4. If you want OpenAI-generated cache/live text, add `OPENAI_API_KEY` in the
   Render dashboard.
5. Deploy.

If `OPENAI_API_KEY` is omitted or rate-limited, the app still generates cached
fallback text locally so the deployed site remains usable.

For low-memory Render instances, the default deployment now uses a smaller
startup workload:

- `RENDER_SAMPLE_TRANSACTIONS=50000`
- `RENDER_CACHE_USERS=3`
- `RENDER_TRAIN_FACTORS=16`
- `RENDER_TRAIN_EPOCHS=4`

## Repository Structure

```
hm-recommender/
├── app/
│   ├── main.py                  # Streamlit entry point
│   └── pages/
│       ├── profile.py           # Customer profile view
│       └── recommendations.py   # Recommendations grid
├── data/
│   ├── EDA.ipynb                # Exploratory data analysis
│   ├── raw/                     # Raw H&M CSVs downloaded from the public mirror
│   └── processed/               # Parquet files (git-ignored)
├── models/
│   ├── als_model.npz            # Trained ALS model
│   ├── article_embeddings.npy   # Sentence-transformer embeddings
│   └── user_item_matrix.npz     # Sparse user-item matrix
├── demo_cache/
│   ├── taste_profiles.json      # Pre-generated taste profiles
│   └── explanations.json        # Pre-generated item explanations
├── legacy/
│   ├── README.md                # Notes on archived application code
│   └── scripts/                 # Previous version of the application
├── src/
│   ├── preprocessing.py         # Data cleaning + feature engineering
│   ├── training.py              # ALS model training
│   ├── content_based.py         # Article embeddings + cosine similarity
│   ├── hybrid.py                # Weighted ALS + content scoring
│   ├── llm.py                   # Taste profile + explanation generation
│   ├── inference.py             # End-to-end pipeline
│   └── evaluation.py            # Precision@K, Recall@K, NDCG@K
├── report/                      # Final PDF report (added Week 3)
├── environment.yml
├── Dockerfile
├── .gitignore
└── LICENSE
```

## Team

| Member  | Role                        |
|---------|-----------------------------|
| Ponnie  | PM + Report Lead            |
| Akash   | Data Engineer               |
| Tianshi | ML Developer A (CF)         |
| Fuwei   | ML Developer B (LLM + Hybrid) |
| Lei     | Frontend + Presentation Lead |

## Retraining the model

```bash
# Re-run after updating data/processed/
python src/training.py --epochs 20 --factors 64 --regularization 0.01
python src/evaluation.py  # prints Precision@10, Recall@10, NDCG@10
```

## Data source note

The downloader uses the public Hugging Face mirror at
`einrafh/hnm-fashion-recommendations-data`, which mirrors the original H&M
recommendation dataset structure (`articles.csv`, `customers.csv`,
`transactions_train.csv`) without requiring a Kaggle login. Use of the data is
still subject to the original dataset terms.

## Running the app

For the full pipeline with public mirrored CSVs and no local image download:

```bash
conda activate hm-recommender
python src/download_data.py --sample-transactions 500000
python src/preprocessing.py
python src/training.py
python src/content_based.py
python src/inference.py --cache
PYTHONPATH=. streamlit run app/main.py
```

If you only want to open the current lightweight demo already supported in this
repo, generate demo artifacts instead:

```bash
conda activate hm-recommender
python src/demo_artifacts.py
PYTHONPATH=. streamlit run app/main.py
```

## Success criteria

- **Recognition test**: ≥4/5 testers say "yes, that's me" about their taste profile.
- **Visible coherence**: recommended items visually cluster around the stated style.
- **Grounded explanations**: each LLM explanation references ≥1 concrete attribute (color, category, silhouette).
- **Demo robustness**: runs end-to-end offline on the curated customer set.
