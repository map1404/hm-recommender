FROM python:3.10-slim

LABEL maintainer="H&M Recommender Team"
LABEL description="H&M User Taste Profile Recommender — Streamlit app"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate synthetic demo artifacts so the app works without a dataset
RUN PYTHONPATH=. python src/demo_artifacts.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

ENTRYPOINT ["streamlit", "run", "app/main.py", \
    "--server.port=8501", "--server.address=0.0.0.0"]
