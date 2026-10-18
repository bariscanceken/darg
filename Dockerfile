# DARG API (FastAPI + hybrid RAG). CPU only: the large language model runs at OpenAI, not on this server.
# The two small retrieval models are baked into the image; the corpus, index and app database are mounted
# from the host (see docker-compose.yml and deploy/KURULUM.md).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU build of PyTorch (the default wheel would pull ~2 GB of CUDA libraries that this server never uses).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Retrieval models, downloaded once at build time; at run time they load with local_files_only.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('intfloat/multilingual-e5-small'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

COPY src ./src
COPY data/geo ./data/geo
COPY data/province_cards ./data/province_cards
COPY data/query_normalization ./data/query_normalization

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/saglik', timeout=4)"

# One worker: every worker loads the models and the index again (~3 GB each).
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
