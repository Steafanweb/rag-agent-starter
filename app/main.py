main.pyimport asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.database import Document, clear_vectors, get_session, init_db
from app.rag import RAGEngine


_rag: Optional[RAGEngine] = None


def get_rag() -> RAGEngine:
    if _rag is None:
        raise RuntimeError("RAGEngine non initialise.")
    return _rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag
    for attempt in range(10):
        try:
            init_db()
            _rag = RAGEngine()
            print("Base de donnees et RAGEngine prets.")
            break
        except Exception as exc:
            if attempt == 9:
                raise RuntimeError(f"Impossible de se connecter apres 10 tentatives: {exc}") from exc
            print(f"DB pas encore prete (tentative {attempt + 1}/10) — nouvelle tentative dans 2s")
            await asyncio.sleep(2)
    yield


app = FastAPI(
    title="RAG Agent Starter",
    description="Upload des PDFs et interrogez-les en langage naturel.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"application/pdf", "text/plain", "text/markdown"}
MAX_MB = 10


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


@app.get("/", tags=["Info"])
def root(session: Session = Depends(get_session)):
    doc_count = session.query(Document).count()
    return {
        "status": "ok",
        "version": "2.0.0",
        "storage": "pgvector (persistant)",
        "documents_indexed": doc_count,
    }


@app.post("/upload", tags=["Documents"])
async def upload(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    rag: RAGEngine = Depends(get_rag),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Type non supporte: {file.content_type}")
    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"Fichier trop lourd (max {MAX_MB} Mo).")
    try:
        result = rag.add_document(content, file.filename)
        doc = Document(filename=file.filename, chunks=result["chunks"])
        session.add(doc)
        session.commit()
        return {"status": "indexe", **result}
    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"Erreur lors de l'indexation: {e}")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    body: QueryRequest,
    session: Session = Depends(get_session),
    rag: RAGEngine = Depends(get_rag),
):
    if not body.question.strip():
        raise HTTPException(400, "La question ne peut pas etre vide.")
    if session.query(Document).count() == 0:
        raise HTTPException(400, "Aucun document indexe.")
    try:
        return rag.query(body.question)
    except Exception as e:
        raise HTTPException(500, f"Erreur lors de la requete: {e}")


@app.get("/documents", tags=["Documents"])
def list_documents(session: Session = Depends(get_session)):
    docs = session.query(Document).order_by(Document.uploaded_at.desc()).all()
    return {
        "count": len(docs),
        "documents": [
            {"id": d.id, "filename": d.filename, "chunks": d.chunks, "uploaded_at": d.uploaded_at.isoformat()}
            for d in docs
        ],
    }


@app.delete("/documents", tags=["Documents"])
def clear_documents(
    session: Session = Depends(get_session),
    rag: RAGEngine = Depends(get_rag),
):
    clear_vectors()
    session.execute(delete(Document))
    session.commit()
    return {"status": "index vide"}
