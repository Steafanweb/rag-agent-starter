# RAG Agent Starter

API REST qui permet d'**uploader des documents PDF** et de les **interroger en langage naturel**.  
Chaque réponse cite sa source, son numéro de page et un extrait du passage utilisé.

Construit avec **FastAPI · LlamaIndex · Claude API (Anthropic)**.

---

## Ce que ça fait

```
POST /upload   →  Indexe un PDF ou fichier texte
POST /query    →  Pose une question → réponse + sources citées
GET  /documents →  Liste les documents indexés
DELETE /documents → Vide l'index
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

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/steafanweb/rag-agent-starter.git
cd rag-agent-starter

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API
cp .env.example .env
# Édite .env et mets ta clé Anthropic

# 5. Lancer l'API
uvicorn app.main:app --reload
```

L'API est disponible sur `http://localhost:8000`  
Documentation interactive : `http://localhost:8000/docs`

---

## Stack technique

| Composant | Technologie |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration RAG | LlamaIndex Core |
| LLM | Claude Sonnet (Anthropic) |
| Embeddings | BAAI/bge-small-en-v1.5 (local, gratuit) |
| Lecture PDF | pypdf |
| Stockage index | Mémoire (in-memory) |

> Les embeddings tournent **localement** — aucune clé API supplémentaire n'est requise.  
> Seule la génération de réponses utilise l'API Anthropic.

---

## Structure du projet

```
rag-agent-starter/
├── app/
│   ├── main.py        # Endpoints FastAPI
│   └── rag.py         # Moteur RAG (LlamaIndex)
├── requirements.txt
├── .env.example       # Template de configuration
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

## Limitation

L'index est **en mémoire** — redémarrer l'API efface les documents indexés.  
Pour une version persistante, voir les intégrations LlamaIndex avec PostgreSQL + pgvector.

---

## Auteur

**Mustapha Ouichka** — Développeur Full Stack · IA & Automatisation  
[Portfolio](https://steafanweb.github.io) · [LinkedIn](https://www.linkedin.com/in/mustapha-ouichka-5a317a320/)

Ce projet est une version open source épurée de l'**Assistant Documentaire** développé en production.
