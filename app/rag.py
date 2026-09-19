import logging
import os
import tempfile
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.postgres import PGVectorStore

logger = logging.getLogger(__name__)

EMBED_DIM = 384  # BAAI/bge-small-en-v1.5


class RAGEngine:
    """
    Moteur RAG avec persistance pgvector.
    Les embeddings survivent aux redémarrages — aucune ré-indexation requise.
    """

    def __init__(self) -> None:
        Settings.llm = Anthropic(
            model="claude-sonnet-4-6",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=2048,
        )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        self._store = PGVectorStore.from_params(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "ragdb"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            table_name="rag_vectors",
            embed_dim=EMBED_DIM,
        )

        self._index = VectorStoreIndex(
            [],
            storage_context=StorageContext.from_defaults(vector_store=self._store),
        )

    # ── Indexation ──────────────────────────────────────────────────────────

    def add_document(self, content: bytes, filename: str) -> dict:
        """
        Indexe un document et retourne ses ref_doc_ids LlamaIndex.
        Ces IDs sont nécessaires pour pouvoir supprimer le document plus tard.
        """
        suffix = Path(filename).suffix or ".pdf"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            docs = SimpleDirectoryReader(input_files=[tmp_path]).load_data()
            ref_doc_ids = []

            for doc in docs:
                doc.metadata["filename"] = filename
                self._index.insert(doc)
                ref_doc_ids.append(doc.doc_id)  # Fix 4 : on collecte les IDs

            logger.info("Indexé '%s' → %d chunks, %d ref_doc_ids.", filename, len(docs), len(ref_doc_ids))
            return {"filename": filename, "chunks": len(docs), "ref_doc_ids": ref_doc_ids}
        finally:
            os.unlink(tmp_path)

    # ── Requête ─────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """Interroge pgvector et génère une réponse avec Claude."""
        engine = self._index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact",
        )
        response = engine.query(question)

        sources = []
        for node in response.source_nodes:
            sources.append({
                "filename": node.metadata.get("filename", "inconnu"),
                "page":     node.metadata.get("page_label", "N/A"),
                "score":    round(node.score, 3) if node.score else None,
                "excerpt":  node.text[:250] + "…" if len(node.text) > 250 else node.text,
            })

        return {"answer": str(response), "sources": sources}

    # ── Suppression unitaire (Fix 4) ─────────────────────────────────────────

    def delete_document(self, ref_doc_ids: list[str]) -> None:
        """Supprime les vecteurs d'un seul document via ses ref_doc_ids LlamaIndex."""
        for ref_doc_id in ref_doc_ids:
            self._index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)
        logger.info("Supprimé %d chunks pour ref_doc_ids %s.", len(ref_doc_ids), ref_doc_ids[:3])

    # ── Nettoyage total (Fix 8) ──────────────────────────────────────────────

    def clear(self) -> None:
        """
        Vide tous les vecteurs via l'API LlamaIndex PGVectorStore.
        Préféré à DELETE SQL direct car il reste valide si LlamaIndex
        change son schéma interne dans une future version.
        """
        try:
            self._store.clear()
            logger.info("pgvector vidé via PGVectorStore.clear().")
        except Exception as exc:
            logger.error("PGVectorStore.clear() échoué : %s — relance clear_vectors() SQL.", exc, exc_info=True)
            raise
