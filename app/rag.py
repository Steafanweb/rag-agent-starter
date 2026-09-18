rag.pyimport os
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

EMBED_DIM = 384


class RAGEngine:
    """Moteur RAG avec persistance pgvector."""

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

    def add_document(self, content: bytes, filename: str) -> dict:
        suffix = Path(filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            docs = SimpleDirectoryReader(input_files=[tmp_path]).load_data()
            for doc in docs:
                doc.metadata["filename"] = filename
            for doc in docs:
                self._index.insert(doc)
            return {"filename": filename, "chunks": len(docs)}
        finally:
            os.unlink(tmp_path)

    def query(self, question: str) -> dict:
        engine = self._index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact",
        )
        response = engine.query(question)
        sources = []
        for node in response.source_nodes:
            sources.append({
                "filename": node.metadata.get("filename", "inconnu"),
                "page": node.metadata.get("page_label", "N/A"),
                "score": round(node.score, 3) if node.score else None,
                "excerpt": node.text[:250] + "..." if len(node.text) > 250 else node.text,
            })
        return {"answer": str(response), "sources": sources}
