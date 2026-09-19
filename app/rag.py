import logging
import os
import tempfile
from pathlib import Path

from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.postgres import PGVectorStore

logger = logging.getLogger(__name__)

# FIX 4 — Modèle multilingue (FR · AR · EN) en remplacement du modèle anglais uniquement.
# paraphrase-multilingual-MiniLM-L12-v2 conserve la même dimension (384) :
# aucune migration de schéma pgvector n'est nécessaire, mais les vecteurs existants
# indexés avec bge-small-en-v1.5 doivent être supprimés (DELETE /documents) puis
# ré-indexés pour garantir la cohérence sémantique.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384

# FIX 3 — Seuil de pertinence : tout passage en dessous est exclu avant la synthèse.
MIN_RELEVANCE_SCORE = 0.45

# FIX 3 — Prompt système strict : Claude refuse de répondre si le contexte est insuffisant.
STRICT_QA_TEMPLATE = PromptTemplate(
    "Tu es un assistant documentaire strict.\n"
    "Réponds UNIQUEMENT en te basant sur les extraits fournis ci-dessous.\n"
    "Si les extraits ne contiennent pas la réponse, réponds EXACTEMENT :\n"
    "\"Je ne trouve pas cette information dans les documents fournis.\"\n"
    "N'invente jamais de données, de chiffres ou de faits.\n\n"
    "Extraits pertinents :\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question : {query_str}\n"
    "Réponse : "
)


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
        # FIX 4 — Modèle multilingue
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

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
                ref_doc_ids.append(doc.doc_id)

            logger.info(
                "Indexé '%s' → %d chunks, %d ref_doc_ids.",
                filename,
                len(docs),
                len(ref_doc_ids),
            )
            return {"filename": filename, "chunks": len(docs), "ref_doc_ids": ref_doc_ids}
        finally:
            os.unlink(tmp_path)

    # ── Requête ─────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Interroge pgvector et génère une réponse avec Claude.

        FIX 3 — Double protection anti-hallucination :
        1. SimilarityPostprocessor filtre les passages en dessous du seuil
           AVANT que Claude ne les reçoive.
        2. STRICT_QA_TEMPLATE interdit à Claude de répondre hors-contexte.
        """
        engine = self._index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=MIN_RELEVANCE_SCORE)
            ],
            text_qa_template=STRICT_QA_TEMPLATE,
        )
        response = engine.query(question)

        sources = []
        for node in response.source_nodes:
            sources.append({
                "filename": node.metadata.get("filename", "inconnu"),
                "page": node.metadata.get("page_label", "N/A"),
                "score": round(node.score, 3) if node.score else None,
                "excerpt": node.text[:250] + "…" if len(node.text) > 250 else node.text,
            })

        answer = str(response)

        # Si tous les passages ont été filtrés (score trop bas) ou réponse vide
        if not sources or answer.strip() in ("Empty Response", ""):
            return {
                "answer": "Je ne trouve pas cette information dans les documents fournis.",
                "sources": [],
            }

        return {"answer": answer, "sources": sources}

    # ── Suppression unitaire ─────────────────────────────────────────────────

    def delete_document(self, ref_doc_ids: list[str]) -> None:
        """Supprime les vecteurs d'un seul document via ses ref_doc_ids LlamaIndex."""
        for ref_doc_id in ref_doc_ids:
            self._index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)
        logger.info("Supprimé %d chunks pour ref_doc_ids %s.", len(ref_doc_ids), ref_doc_ids[:3])

    # ── Nettoyage total ──────────────────────────────────────────────────────

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
            logger.error(
                "PGVectorStore.clear() échoué : %s — relance clear_vectors() SQL.", exc, exc_info=True
            )
            raise
