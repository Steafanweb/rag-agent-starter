# RAG Agent Starter

API REST qui permet d'**uploader des documents PDF** et de les **interroger en langage naturel**.  
Chaque réponse cite sa source, son numéro de page et un extrait du passage utilisé.

Construit avec **FastAPI · LlamaIndex · Claude API · pgvector**.

---

## Ce que ça fait

```
POST /upload      →  Indexe un PDF ou fichier texte
POST /query       →  Pose une question → réponse + sources citées
GET  /documents   →  Liste les documents indexés
DELETE /documents →  Vide l'index
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
  "answer": "Le chiffre d'affaires 2024 est de 4,2 millions de dinars, en hausse de 12%.",
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

## Démarrage rapide — Docker Compose ✅ (recommandé)

C'est la méthode la plus simple : une commande lance PostgreSQL/pgvector et l'API ensemble.

```bash
# 1. Cloner le repo
git clone https://github.com/steafanweb/rag-agent-starter.git
cd rag-agent-starter

# 2. Configurer les variables d'environnement
cp .env.example .env
# Édite .env et renseigne ta clé Anthropic (et change POSTGRES_PASSWORD)

# 3. Lancer
docker compose up --build
```

L'API est disponible sur `http://localhost:8000`  
Documentation interactive : `http://localhost:8000/docs`

Pour stopper sans perdre les données :
```bash
docker compose stop       # arrête les conteneurs, conserve le volume pgdata
docker compose down       # arrête ET supprime les conteneurs (volume conservé)
docker compose down -v    # ⚠️  supprime aussi le volume → perte des données
```

---

## Installation locale (sans Docker)

```bash
# 1. Prérequis : PostgreSQL avec extension pgvector installée
#    https://github.com/pgvector/pgvector#installation

# 2. Cloner et créer l'environnement virtuel
git clone https://github.com/steafanweb/rag-agent-starter.git
cd rag-agent-starter
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer
cp .env.example .env
# Édite .env avec ta clé Anthropic et les infos PostgreSQL locales

# 5. Lancer
uvicorn app.main:app --reload
```

---

## Persistance des données

| | v1 (in-memory) | v2 (pgvector) |
|---|---|---|
| Stockage | Mémoire RAM | PostgreSQL + pgvector |
| Survie au redémarrage | ❌ | ✅ |
| Scalabilité | Mono-instance | Multi-instance possible |
| Setup | `pip install` | Docker Compose ou PostgreSQL local |

Les embeddings sont stockés dans la table `data_rag_vectors` (gérée par LlamaIndex).  
Le registre des documents (métadonnées) est dans `rag_documents` (gérée par SQLAlchemy).

---

## Stack technique

| Composant | Technologie |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration RAG | LlamaIndex Core |
| LLM | Claude Sonnet (Anthropic) |
| Embeddings | BAAI/bge-small-en-v1.5 (local, gratuit) |
| Stockage vecteurs | pgvector (PostgreSQL) |
| Tracking documents | SQLAlchemy + PostgreSQL |
| Lecture PDF | pypdf |
| Conteneurisation | Docker + Docker Compose |

> Les embeddings tournent **localement** — aucune clé API supplémentaire n'est requise.  
> Seule la génération de réponses utilise l'API Anthropic.

---

## Structure du projet

```
rag-agent-starter/
├── app/
│   ├── main.py        # Endpoints FastAPI
│   ├── rag.py         # Moteur RAG (LlamaIndex + pgvector)
│   └── database.py    # SQLAlchemy — connexion, modèles, init
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Formats supportés

- **PDF** (`.pdf`)
- **Texte** (`.txt`)
- **Markdown** (`.md`)

Taille maximale : 10 Mo par fichier.

---

## Extension : index persistant avec pgvector

Pour une version encore plus robuste (multi-instance, backup, index sur disque),
voir les intégrations LlamaIndex avec `PGVectorStore` + connexion pool :
https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/

---

## Auteur

**Mustapha Ouichka** — Développeur Full Stack · IA & Automatisation  
[Portfolio](https://steafanweb.github.io) · [LinkedIn](https://www.linkedin.com/in/mustapha-ouichka-5a317a320/)

Ce projet est une version open source épurée de l'**Assistant Documentaire** développé en production.
