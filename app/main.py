# FIX 2 — load_dotenv() doit être appelé avant tout import qui lit des variables
# d'environnement. app.database lit POSTGRES_* au niveau module ; on le charge ici
# en premier pour que uvicorn app.main:app fonctionne sans export préalable.
from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.database import Document, SessionLocal, clear_vectors, get_session, init_db
from app.rag import RAGEngine

logger = logging.getLogger(__name__)

# ── Rate limiting (Fix 2) ────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── RAGEngine ────────────────────────────────────────────────────────────────

_rag: Optional[RAGEngine] = None


def get_rag() -> RAGEngine:
    if _rag is None:
        raise RuntimeError("RAGEngine non initialisé.")
    return _rag


# ── Démarrage ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag

    # FIX 4 — Vérification de la clé API au démarrage.
    # En production (ENVIRONMENT=production), l'absence de API_KEY est une erreur fatale.
    # En développement, on lève un avertissement visible mais on continue.
    api_key = os.getenv("API_KEY", "").strip()
    env = os.getenv("ENVIRONMENT", "dev").lower()
    if not api_key:
        if env == "production":
            raise RuntimeError(
                "API_KEY doit être définie en production. "
                "Définissez-la dans .env ou en variable d'environnement."
            )
        logger.warning(
            "⚠️  API_KEY non définie — authentification désactivée. "
            "Acceptable uniquement en développement local."
        )

    # FIX 6 — Vérification de la robustesse du mot de passe PostgreSQL au démarrage.
    pg_password = os.getenv("POSTGRES_PASSWORD", "")
    if len(pg_password) < 16:
        msg = (
            f"POSTGRES_PASSWORD trop court ({len(pg_password)} car.) — minimum 16 caractères requis."
        )
        if env == "production":
            raise RuntimeError(msg)
        logger.warning("⚠️  %s Acceptable uniquement en développement local.", msg)

    for attempt in range(10):
        try:
            init_db()
            _rag = RAGEngine()
            logger.info("Base de données et RAGEngine prêts.")
            break
        except Exception as exc:
            if attempt == 9:
                raise RuntimeError(f"Impossible de se connecter à PostgreSQL : {exc}") from exc
            logger.warning("DB pas encore prête (tentative %d/10) — retry dans 2s", attempt + 1)
            await asyncio.sleep(2)
    yield


# ── Application ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Agent Starter",
    description="Upload des PDFs · Interrogez en langage naturel · Citations sources et pages.",
    version="3.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# FIX 8 — CORS restreint aux origines explicitement autorisées.
# Définissez CORS_ORIGINS dans .env pour la production (ex. : https://mon-app.exemple.com).
# Valeur par défaut : localhost uniquement (développement).
_cors_origins = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost,http://localhost:8000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"application/pdf", "text/plain", "text/markdown"}
MAX_MB = 10
_UI_PATH = Path(__file__).parent.parent / "interface.html"


# ── Schémas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    filename: str
    page: str
    score: float | None
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root(
    session: Session = Depends(get_session),
    _: None = Depends(verify_api_key),           # Fix 1
):
    doc_count = session.query(Document).count()
    return {
        "status": "ok",
        "version": "3.0.0",
        "storage": "pgvector (persistant)",
        "documents_indexed": doc_count,
    }


@app.get("/ui", include_in_schema=False)         # Fix 10 : interface servie par FastAPI
async def serve_ui():
    if not _UI_PATH.exists():
        raise HTTPException(404, "Interface non disponible.")
    return FileResponse(_UI_PATH)


@app.post("/upload", tags=["Documents"])
@limiter.limit("20/minute")                      # Fix 2
async def upload(
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    rag: RAGEngine = Depends(get_rag),
    _: None = Depends(verify_api_key),           # Fix 1
):
    """
    Upload et indexe un document. Rejette les doublons.
    Les vecteurs persistent dans pgvector après redémarrage.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Type non supporté : {file.content_type}.")

    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"Fichier trop lourd (max {MAX_MB} Mo).")

    # Fix 3 : déduplication — rejette si le fichier est déjà indexé
    if session.query(Document).filter(Document.filename == file.filename).first():
        raise HTTPException(409, f"'{file.filename}' est déjà indexé. Supprimez-le d'abord.")

    try:
        # Opération bloquante (embeddings) → thread pool pour ne pas bloquer l'event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, rag.add_document, content, file.filename)

        # Fix 4 : on stocke les ref_doc_ids pour pouvoir supprimer ce doc plus tard
        doc = Document(
            filename=file.filename,
            chunks=result["chunks"],
            ref_doc_ids=result["ref_doc_ids"],
        )
        session.add(doc)
        session.commit()

        return {"status": "indexé", "filename": result["filename"], "chunks": result["chunks"]}
    except HTTPException:
        raise
    except Exception as exc:
        session.rollback()
        logger.error("Erreur indexation '%s' : %s", file.filename, exc, exc_info=True)
        raise HTTPException(500, "Erreur interne lors de l'indexation.")   # Fix 6


@app.post("/query", response_model=QueryResponse, tags=["Query"])
@limiter.limit("30/minute")                      # Fix 2
async def query(
    request: Request,
    body: QueryRequest,
    session: Session = Depends(get_session),
    rag: RAGEngine = Depends(get_rag),
    _: None = Depends(verify_api_key),           # Fix 1
):
    if not body.question.strip():
        raise HTTPException(400, "La question ne peut pas être vide.")

    if session.query(Document).count() == 0:
        raise HTTPException(400, "Aucun document indexé. Uploadez au moins un fichier d'abord.")

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, rag.query, body.question)
    except Exception as exc:
        logger.error("Erreur requête '%s' : %s", body.question, exc, exc_info=True)
        raise HTTPException(500, "Erreur interne lors de la requête.")     # Fix 6


@app.get("/documents", tags=["Documents"])
def list_documents(
    session: Session = Depends(get_session),
    _: None = Depends(verify_api_key),           # Fix 1
    limit: int = Query(default=20, ge=1, le=100),  # Fix 9 : pagination
    offset: int = Query(default=0, ge=0),
):
    """Liste les documents indexés avec pagination (limit/offset)."""
    total = session.query(Document).count()
    docs = (
        session.query(Document)
        .order_by(Document.uploaded_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {
        "total":     total,
        "limit":     limit,
        "offset":    offset,
        "documents": [
            {
                "id":          d.id,
                "filename":    d.filename,
                "chunks":      d.chunks,
                "uploaded_at": d.uploaded_at.isoformat(),
            }
            for d in docs
        ],
    }


@app.delete("/documents/{doc_id}", tags=["Documents"])   # Fix 4 : suppression unitaire
def delete_one_document(
    doc_id: int,
    session: Session = Depends(get_session),
    rag: RAGEngine = Depends(get_rag),
    _: None = Depends(verify_api_key),
):
    """Supprime un seul document de l'index (vecteurs + registre)."""
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document introuvable.")

    try:
        rag.delete_document(doc.ref_doc_ids or [])
        session.delete(doc)
        session.commit()
        return {"status": "supprimé", "filename": doc.filename}
    except Exception as exc:
        session.rollback()
        logger.error("Erreur suppression doc %d : %s", doc_id, exc, exc_info=True)
        raise HTTPException(500, "Erreur interne lors de la suppression.")


@app.delete("/documents", tags=["Documents"])
def clear_all_documents(
    session: Session = Depends(get_session),
    rag: RAGEngine = Depends(get_rag),
    _: None = Depends(verify_api_key),
):
    """Supprime TOUS les documents de l'index pgvector et du registre."""
    try:
        rag.clear()
    except Exception:
        # Fallback : DELETE SQL direct si PGVectorStore.clear() échoue
        clear_vectors()
    session.execute(delete(Document))
    session.commit()
    return {"status": "index vidé"}
