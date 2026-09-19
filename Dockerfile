FROM python:3.11-slim

WORKDIR /app

# Dépendances système : psycopg2 (libpq) + compilation C
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-télécharge le modèle d'embeddings au build → évite 30–60s de cold start
# FIX — modèle multilingue aligné sur app/rag.py (FR · AR · EN, dim 384).
# ⚠️  Remplace l'ancien bge-small-en-v1.5 (anglais uniquement).
RUN python -c "\
from llama_index.embeddings.huggingface import HuggingFaceEmbedding; \
HuggingFaceEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
