"""
Tests basiques de l'API RAG Agent Starter.
Lance avec : pytest tests/ -v
"""
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

# Variables d'env AVANT l'import de l'app
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_DB", "ragdb")
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")

HEADERS = {"X-API-Key": "test-key"}


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """
    Client de test avec DB et RAGEngine mockés.
    On patche AVANT l'import pour éviter toute connexion réelle.
    """
    mock_rag = MagicMock()
    mock_rag.add_document.return_value = {
        "filename": "test.pdf",
        "chunks": 3,
        "ref_doc_ids": ["id-1", "id-2", "id-3"],
    }
    mock_rag.query.return_value = {
        "answer": "La réponse est 42.",
        "sources": [{"filename": "test.pdf", "page": "1", "score": 0.95, "excerpt": "…"}],
    }

    # Session SQLAlchemy mockée
    mock_doc = MagicMock()
    mock_doc.id = 1
    mock_doc.filename = "test.pdf"
    mock_doc.chunks = 3
    mock_doc.ref_doc_ids = ["id-1", "id-2", "id-3"]
    mock_doc.uploaded_at.isoformat.return_value = "2026-09-19T10:00:00+00:00"

    mock_session = MagicMock()
    mock_session.__enter__ = lambda s: s
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.query.return_value.count.return_value = 0
    mock_session.query.return_value.filter.return_value.first.return_value = None
    mock_session.query.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []

    with (
        patch("app.database.create_engine"),
        patch("app.rag.PGVectorStore"),
        patch("app.rag.HuggingFaceEmbedding"),
        patch("app.rag.Anthropic"),
        patch("app.main.init_db"),
        patch("app.main.RAGEngine", return_value=mock_rag),
    ):
        from fastapi.testclient import TestClient
        from app.main import app, get_rag, get_session

        # FIX fixture — get_session est un générateur (yield), l'override doit l'être aussi.
        # lambda: iter([mock_session]) transmettait l'itérateur lui-même comme "session" ;
        # FastAPI recevait un objet iter au lieu de mock_session.
        def _mock_get_session():
            yield mock_session

        app.dependency_overrides[get_session] = _mock_get_session
        app.dependency_overrides[get_rag] = lambda: mock_rag

        with TestClient(app) as c:
            yield c


# ── Tests auth (Fix 1) ───────────────────────────────────────────────────────

def test_no_api_key_returns_403(client):
    """Sans header X-API-Key → 403."""
    r = client.get("/")
    assert r.status_code == 403

def test_wrong_api_key_returns_403(client):
    """Mauvaise clé → 403."""
    r = client.get("/", headers={"X-API-Key": "mauvaise-cle"})
    assert r.status_code == 403

def test_valid_api_key_returns_200(client):
    """Bonne clé → 200."""
    r = client.get("/", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Tests upload (Fix 3 déduplication) ──────────────────────────────────────

def test_upload_invalid_content_type(client):
    """Fichier CSV non supporté → 400."""
    r = client.post(
        "/upload",
        headers=HEADERS,
        files={"file": ("data.csv", BytesIO(b"a,b,c"), "text/csv")},
    )
    assert r.status_code == 400

def test_upload_too_large(client):
    """Fichier > 10 Mo → 400."""
    r = client.post(
        "/upload",
        headers=HEADERS,
        files={"file": ("big.pdf", BytesIO(b"x" * (11 * 1024 * 1024)), "application/pdf")},
    )
    assert r.status_code == 400


# ── Tests query ──────────────────────────────────────────────────────────────

def test_query_empty_question(client):
    """Question vide → 400."""
    r = client.post("/query", headers=HEADERS, json={"question": ""})
    assert r.status_code == 400

def test_query_whitespace_question(client):
    """Question avec que des espaces → 400."""
    r = client.post("/query", headers=HEADERS, json={"question": "   "})
    assert r.status_code == 400

def test_query_no_documents_returns_400(client):
    """Aucun document indexé → 400 avec message clair."""
    r = client.post("/query", headers=HEADERS, json={"question": "Quelle est la réponse ?"})
    assert r.status_code == 400
    assert "document" in r.json()["detail"].lower()


# ── Tests /documents (Fix 9 pagination) ─────────────────────────────────────

def test_list_documents_has_pagination_fields(client):
    """GET /documents doit retourner total, limit, offset."""
    r = client.get("/documents", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert "total" in data
    assert "limit" in data
    assert "offset" in data
    assert "documents" in data

def test_list_documents_limit_param(client):
    """Le paramètre limit est accepté."""
    r = client.get("/documents?limit=5&offset=0", headers=HEADERS)
    assert r.status_code == 200

def test_list_documents_invalid_limit(client):
    """limit=0 → 422 (validation Pydantic)."""
    r = client.get("/documents?limit=0", headers=HEADERS)
    assert r.status_code == 422


# ── Test interface (Fix 10) ──────────────────────────────────────────────────

def test_ui_endpoint_accessible(client):
    """/ui accessible sans auth (page statique publique)."""
    r = client.get("/ui")
    assert r.status_code in (200, 404)  # 404 si interface.html absent en test
