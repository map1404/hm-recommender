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
