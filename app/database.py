import os
  from datetime import datetime, timezone

    from sqlalchemy import Column, DateTime, Integer, String, create_engine, text
      from sqlalchemy.orm import DeclarativeBase, sessionmaker

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
            """Registre des documents uploadés (métadonnées uniquement)."""

            __tablename__ = "rag_documents"

            id = Column(Integer, primary_key=True, autoincrement=True)
            filename = Column(String(512), nullable=False)
            chunks = Column(Integer, nullable=False, default=0)
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


      # ── Helpers ──────────────────────────────────────────────────────────────────

      def clear_vectors() -> None:
    """
          Supprime tous les vecteurs de la table interne LlamaIndex pgvector.
          LlamaIndex nomme la table 'data_{table_name}', soit 'data_rag_vectors' ici.
                """
                with engine.connect() as conn:
                    try:
                        conn.execute(text("DELETE FROM data_rag_vectors"))
                        conn.commit()
                    except Exception:
            conn.rollback()  # La table n'existe pas encore — aucun problème
