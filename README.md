# RAG Agent Starter

API REST pour **uploader des documents** et les **interroger en langage naturel**.  
Chaque réponse cite sa source, son numéro de page et un extrait du passage utilisé.

Construit avec **FastAPI · LlamaIndex · Claude API · pgvector**.

---

## Ce que ça fait

```
GET    /            →  Statut de l'API + nombre de documents indexés
GET    /ui          →  Interface web intégrée
POST   /upload      →  Indexe un PDF, TXT ou Markdown
POST   /query       →  Pose une question → réponse + sources citées
GET    /documents   →  Liste les documents indexés (paginé)
DELETE /documents/{id}  →  Supprime un document précis
DELETE /documents   →  Vide l'index complet
```

**Exemple :**

```bash
# 1. Uploader un document
curl -X POST http://localhost:8000/upload \
  -H "X-API-Key: votre_cle_api" \
  -F "file=@rapport_annuel.pdf"

# 2. Poser une question
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: votre_cle_api" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quel est le chiffre d affaires 2024 ?"}'
```

**Réponse :**
```json
{
  "answer": "Le chiffre d'affaires 2024 est de 4,2 millions de dinars, en hausse de 12 %.",
  "sources": [
    {
      "filename": "rapport_annuel.pdf",
      "page": "8",
      "score": 0.921,
      "excerpt": "...le CA consolidé atteint 4,2 M TND au 31/12/2024..."
    }
  ]
}
```

---

## Démarrage rapide — Docker Compose (recommandé)

```bash
# 1. Cloner le repo
git clone https://github.com/steafanweb/rag-agent-starter.git
cd rag-agent-starter

# 2. Créer le fichier d'environnement
cp .env.example .env
```

Édite `.env` et renseigne **au minimum** ces trois variables :

```env
ANTHROPIC_API_KEY=sk-ant-api03-...   # clé Anthropic réelle
API_KEY=<générer avec secrets.token_urlsafe(32)>
POSTGRES_PASSWORD=<générer avec secrets.token_urlsafe(32)>
```

> Commande de génération : `python -c "import secrets; print(secrets.token_urlsafe(32))"`  
> À exécuter deux fois — une pour `API_KEY`, une pour `POSTGRES_PASSWORD`.

```bash
# 3. Lancer
docker compose up --build
```

| URL | Description |
|-----|-------------|
| `http://localhost:8000/ui` | Interface web |
| `http://localhost:8000/docs` | Documentation Swagger interactive |
| `http://localhost:8000` | Statut JSON de l'API |

**Arrêt :**
```bash
docker compose stop       # arrête, conserve les données
docker compose down       # arrête et supprime les conteneurs (données conservées)
docker compose down -v    # ⚠️  supprime aussi le volume → perte des données
```

---

## Installation locale (sans Docker)

Prérequis : PostgreSQL avec l'extension `pgvector` installée  
→ https://github.com/pgvector/pgvector#installation

```bash
git clone https://github.com/steafanweb/rag-agent-starter.git
cd rag-agent-starter
python -m venv venv
source venv/bin/activate       # Linux / Mac
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
cp .env.example .env
# Édite .env : ANTHROPIC_API_KEY, API_KEY, POSTGRES_PASSWORD + connexion locale

uvicorn app.main:app --reload
```

---

## Variables d'environnement

| Variable | Obligatoire | Description |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | ✅ Toujours | Clé API Anthropic |
| `API_KEY` | ✅ En production | Clé d'authentification pour toutes les routes |
| `POSTGRES_PASSWORD` | ✅ Toujours | Mot de passe PostgreSQL (≥ 16 caractères) |
| `POSTGRES_DB` | Non | Nom de la base (défaut : `ragdb`) |
| `POSTGRES_USER` | Non | Utilisateur PostgreSQL (défaut : `raguser`) |
| `ENVIRONMENT` | Non | `production` pour rendre `API_KEY` obligatoire au démarrage |
| `CORS_ORIGINS` | Non | Origines autorisées, séparées par des virgules (défaut : localhost) |

En développement local sans `API_KEY`, un avertissement est affiché mais l'API démarre.  
En production (`ENVIRONMENT=production`), l'absence de `API_KEY` ou un `POSTGRES_PASSWORD` trop court fait échouer le démarrage immédiatement.

---

## Authentification

Toutes les routes sont protégées par une clé API transmise dans le header `X-API-Key`.

```bash
curl -H "X-API-Key: votre_cle_api" http://localhost:8000/documents
```

Sans clé (ou avec une clé incorrecte) → `403 Forbidden`.  
Si `API_KEY` est vide dans `.env`, l'authentification est désactivée (dev local uniquement).

---

## Sécurité

- **Authentification** — clé API sur toutes les routes via `X-API-Key`
- **Rate limiting** — 20 req/min sur `/upload`, 30 req/min sur `/query`
- **Déduplication** — refus d'indexer deux fois le même fichier (`409 Conflict`)
- **Validation des types** — seuls PDF, TXT et Markdown sont acceptés
- **Taille max** — 10 Mo par fichier
- **CORS restreint** — liste d'origines explicite via `CORS_ORIGINS`
- **PostgreSQL isolé** — port lié à `127.0.0.1` uniquement (non exposé au réseau)
- **Mot de passe fort** — démarrage échoue si `POSTGRES_PASSWORD` < 16 caractères en production
- **Pas de secret par défaut** — Docker Compose exige `POSTGRES_PASSWORD` dans `.env`

---

## Tests

```bash
# Tests unitaires (ne nécessitent pas de base de données)
pytest tests/test_api.py -v

# Tests d'intégration (nécessitent le stack Docker complet)
docker compose up -d
pytest tests/test_integration.py -v
```

---

## Persistance des données

| | v1 (in-memory) | v2 (pgvector) |
|---|---|---|
| Stockage | RAM | PostgreSQL + pgvector |
| Survie au redémarrage | ❌ | ✅ |
| Scalabilité | Mono-instance | Multi-instance |
| Setup | `pip install` | Docker Compose ou PostgreSQL local |

Les embeddings sont dans la table `data_rag_vectors` (gérée par LlamaIndex).  
Les métadonnées des documents sont dans `rag_documents` (gérée par SQLAlchemy).

---

## Stack technique

| Composant | Technologie |
|---|---|
| API | FastAPI 0.111 + Uvicorn |
| Orchestration RAG | LlamaIndex Core 0.14.x |
| LLM | Claude Sonnet (Anthropic) |
| Embeddings | BAAI/bge-small-en-v1.5 (local, sans clé API) |
| Stockage vecteurs | pgvector (PostgreSQL 16) |
| Tracking documents | SQLAlchemy 2 + PostgreSQL |
| Lecture PDF | pypdf |
| Rate limiting | SlowAPI |
| Conteneurisation | Docker + Docker Compose |

> Les embeddings tournent **localement** — seule la génération de réponses consomme l'API Anthropic.

---

## Structure du projet

```
rag-agent-starter/
├── app/
│   ├── main.py        # Endpoints FastAPI, lifespan, rate limiting
│   ├── rag.py         # Moteur RAG (LlamaIndex + pgvector)
│   ├── database.py    # SQLAlchemy — connexion, modèles, init
│   └── auth.py        # Vérification de la clé API
├── tests/
│   ├── test_api.py          # Tests unitaires (TestClient)
│   └── test_integration.py  # Tests d'intégration (stack complet)
├── interface.html     # Interface web intégrée
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
└── README.md
```

---

## Formats supportés

| Format | Extension | Taille max |
|--------|-----------|------------|
| PDF | `.pdf` | 10 Mo |
| Texte brut | `.txt` | 10 Mo |
| Markdown | `.md` | 10 Mo |

---

## Auteur

**Mustapha Ouichka** — Développeur Full Stack · IA & Automatisation  
[Portfolio](https://steafanweb.github.io) · [LinkedIn](https://www.linkedin.com/in/mustapha-ouichka-5a317a320/)

Ce projet est une version open source de l'**Assistant Documentaire** développé en production.
