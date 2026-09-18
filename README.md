# RAG Agent Starter

API REST qui permet d'**uploader des documents** et de les **interroger en langage naturel**.  
Chaque réponse cite sa source, son numéro de page et un extrait du passage utilisé.

Construit avec **FastAPI · LlamaIndex · Claude API (Anthropic) · PostgreSQL + pgvector**.

> **v2** — L'index est désormais **persistant** : les documents survivent aux redémarrages grâce à pgvector.

---

## Ce que ça fait

```
POST /upload     →  Indexe un PDF ou fichier texte dans pgvector
POST /query      →  Pose une question → réponse + sources citées
GET  /documents  →  Liste les documents indexés
DELETE /documents →  Vide l'index (vecteurs + métadonnées)
```

**Exemple :**

```bash
# 1. Uploader un document
curl -X POST http://localhost:8000/upload \
  -F "file=@rapport_annuel.pdf"

# 2. Poser une question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quel est le chiffre d affaires 2024 ?"}'
```

**Réponse :**
```json
{
  "answer": "Le chiffre d'affaires 2024 est de 4,2 millions de dinars, en hausse de 12% par rapport à 2023.",
  "sources": [
    {
      "filename": "rapport_annuel.pdf",
      "page": "8",
      "score": 0.921,
      "excerpt": "...le CA consolidé atteint 4,2 M TND au 31/12/2024, soit une progression de 12%..."
    }
  ]
}
```

---

## Démarrage rapide (Docker Compose)

```bash
# 1. Cloner le repo
git clone https://github.com/steafanweb/rag-agent-starter.git
cd rag-agent-starter

# 2. Configurer la clé API
cp .env.example .env
# Édite .env et ajoute ta clé Anthropic :
#   ANTHROPIC_API_KEY=sk-ant-...

# 3. Lancer (PostgreSQL + pgvector + API)
docker compose up --build
```

L'API est disponible sur `http://localhost:8000`  
Documentation interactive : `http://localhost:8000/docs`

---

## Installation locale (sans Docker)

Prérequis : PostgreSQL avec l'extension `pgvector` installée.

```bash
# 1. Environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 2. Dépendances
pip install -r requirements.txt

# 3. Variables d'environnement
cp .env.example .env
# Renseigne ANTHROPIC_API_KEY et les variables POSTGRES_*

# 4. Lancer l'API
uvicorn app.main:app --reload
```

---

## Stack technique

| Composant | Technologie |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration RAG | LlamaIndex Core |
| LLM | Claude Sonnet 4.5 (Anthropic) |
| Embeddings | BAAI/bge-small-en-v1.5 (local, gratuit) |
| Stockage vectoriel | PostgreSQL + pgvector |
| ORM / métadonnées | SQLAlchemy 2.0 |
| Conteneurisation | Docker + Docker Compose |

> Les embeddings tournent **localement** — aucune clé API supplémentaire n'est requise.  
> Seule la génération de réponses utilise l'API Anthropic.

---

## Structure du projet

```
rag-agent-starter/
├── app/
│   ├── __init__.py
│   ├── main.py        # Endpoints FastAPI (lifespan, Depends)
│   ├── rag.py         # Moteur RAG — PGVectorStore + LlamaIndex
│   └── database.py    # SQLAlchemy ORM + init pgvector
├── docker-compose.yml # pgvector/pgvector:pg16 + healthcheck
├── Dockerfile         # Build image + pré-télécharge le modèle d'embeddings
├── requirements.txt
├── interface.html     # Interface web légère (optionnel)
├── .env.example       # Template de configuration
├── .gitignore
└── README.md
```

---

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Obligatoire** — clé API Anthropic |
| `POSTGRES_HOST` | `localhost` | Hôte PostgreSQL |
| `POSTGRES_PORT` | `5432` | Port PostgreSQL |
| `POSTGRES_DB` | `ragdb` | Nom de la base |
| `POSTGRES_USER` | `postgres` | Utilisateur |
| `POSTGRES_PASSWORD` | `postgres` | Mot de passe |

---

## Formats supportés

- **PDF** (`.pdf`)
- **Texte** (`.txt`)
- **Markdown** (`.md`)

---

## Auteur

**Mustapha Ouichka** — Développeur Full Stack · IA & Automatisation  
[Portfolio](https://steafanweb.github.io) · [LinkedIn](https://www.linkedin.com/in/mustapha-ouichka-5a317a320/)

Ce projet est une version open source épurée de l'**Assistant Documentaire** développé en production.
