import logging
import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, JSON, String, UniqueConstraint, create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

# ── Connexion ────────────────────────────────────────────────────────────────

_DB_URL = (
    f"postgresql+psycopg2://"
    f"{os.getenv('POSTGRES_USER', 'postgres')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'ragdb')}"
)

engine = create_engine(_DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


# ── Modèle ───────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class Document(Base):
    """
    Registre des documents uploadés.

    ref_doc_ids : liste des doc_id LlamaIndex pour ce fichier.
                  Nécessaire pour pouvoir supprimer les vecteurs d'un seul document
                  via index.delete_ref_doc().
    """
    __tablename__ = "rag_documents"
    __table_args__ = (
        UniqueConstraint("filename", name="uq_rag_documents_filename"),
    )

    id          = Column(Integer, primary_key=True, autoincrement=True)
    filename    = Column(String(512), nullable=False)
    chunks      = Column(Integer, nullable=False, default=0)
    ref_doc_ids = Column(JSON, nullable=False, default=list)   # Fix 4 : suppression unitaire
    uploaded_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


# ── Init ─────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Active l'extension pgvector et crée les tables si elles n'existent pas."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)


def get_session():
    """Dépendance FastAPI : session SQLAlchemy par requête."""
    with SessionLocal() as session:
        yield session


# ── Helpers vecteurs ─────────────────────────────────────────────────────────

# Fix 8 : nom de table dans une constante → facile à mettre à jour si LlamaIndex change
_VECTOR_TABLE = "data_rag_vectors"


def clear_vectors() -> None:
    """
    Vide tous les vecteurs de la table LlamaIndex pgvector.
    LlamaIndex nomme ses tables 'data_{table_name}', soit 'data_rag_vectors' ici.
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(f"DELETE FROM {_VECTOR_TABLE}"))
            conn.commit()
            logger.info("Table %s vidée avec succès.", _VECTOR_TABLE)
        except Exception as exc:
            conn.rollback()
            logger.error("clear_vectors() échoué sur %s : %s", _VECTOR_TABLE, exc, exc_info=True)
            raise RuntimeError(f"Impossible de vider {_VECTOR_TABLE}.") from exc
