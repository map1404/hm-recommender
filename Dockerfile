FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
RUN pip install --no-cache-dir \
    streamlit pandas numpy scipy implicit sentence-transformers \
    scikit-learn pyarrow Pillow requests tqdm plotly python-dotenv

COPY . .

EXPOSE 8501

HEALTHCHECK CMD /bin/sh -c 'curl --fail "http://localhost:${PORT:-8501}/_stcore/health" || exit 1'

CMD ["/bin/sh", "-c", "python -m src.render_bootstrap && streamlit run app/main.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
