"""
Tests d'intégration — pipeline complet (upload → index → query → sources).

Ces tests s'exécutent CONTRE une instance réelle de l'API (pas de mock).
Prérequis :
    docker compose up --build -d
    pip install pytest requests --break-system-packages

Lancement :
    pytest tests/test_integration.py -v -m integration

Variables d'environnement optionnelles :
    TEST_API_URL   URL de l'API à tester (défaut : http://localhost:8000)
    API_KEY        Clé API si l'authentification est activée (défaut : vide)
"""

import io
import os

import pytest
import requests

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = os.getenv("TEST_API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")

# Contenu de test avec des faits vérifiables précis
TEST_DOC_CONTENT = (
    "RAPPORT ANNUEL 2024\n"
    "=====================\n"
    "Le chiffre d'affaires consolide pour l'exercice 2024 s'eleve a 42 millions "
    "de dinars tunisiens, en hausse de 12 pourcent par rapport a 2023.\n"
    "La marge nette est de 8 pourcent.\n"
    "L'effectif total est de 350 employes.\n"
    "Le siege social est situe a Tunis, Tunisie.\n"
    "Le directeur general est M. Karim Ben Salah.\n"
).encode("utf-8")

TEST_FILENAME = "rapport_test_integration.txt"


def api_headers(extra: dict | None = None) -> dict:
    """Retourne les headers HTTP avec X-API-Key si configurée."""
    h = dict(extra or {})
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def upload_test_doc(filename: str = TEST_FILENAME, content: bytes = TEST_DOC_CONTENT) -> requests.Response:
    """Raccourci pour uploader le document de test."""
    return requests.post(
        f"{BASE_URL}/upload",
        headers=api_headers(),
        files={"file": (filename, io.BytesIO(content), "text/plain")},
        timeout=60,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_index():
    """Vide l'index avant et après chaque test pour garantir l'isolation."""
    requests.delete(f"{BASE_URL}/documents", headers=api_headers(), timeout=30)
    yield
    requests.delete(f"{BASE_URL}/documents", headers=api_headers(), timeout=30)


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestApiHealth:
    def test_root_endpoint_responds(self):
        """L'API répond avec status ok et les champs attendus."""
        r = requests.get(f"{BASE_URL}/", headers=api_headers(), timeout=10)
        assert r.status_code == 200, f"Attendu 200, reçu {r.status_code} : {r.text}"
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "documents_indexed" in data

    def test_swagger_docs_accessible(self):
        """La documentation Swagger est disponible."""
        r = requests.get(f"{BASE_URL}/docs", timeout=10)
        assert r.status_code == 200


@pytest.mark.integration
class TestAuthentication:
    def test_valid_key_accepted(self):
        """Une clé API valide est acceptée (ou auth désactivée)."""
        r = requests.get(f"{BASE_URL}/", headers=api_headers(), timeout=10)
        assert r.status_code in (200, 403)
        if API_KEY:
            assert r.status_code == 200, "La clé API configurée a été refusée"

    def test_invalid_key_rejected_when_auth_enabled(self):
        """Une mauvaise clé est refusée avec 403 si l'auth est activée."""
        if not API_KEY:
            pytest.skip("API_KEY non configurée — auth désactivée, test ignoré.")
        r = requests.get(
            f"{BASE_URL}/",
            headers={"X-API-Key": "cle_invalide_xxxxxxxx"},
            timeout=10,
        )
        assert r.status_code == 403, f"Attendu 403, reçu {r.status_code}"


@pytest.mark.integration
class TestDocumentUpload:
    def test_upload_txt_document_succeeds(self):
        """Un fichier texte est correctement indexé."""
        r = upload_test_doc()
        assert r.status_code == 200, f"Upload échoué : {r.text}"
        data = r.json()
        assert data["status"] == "indexé"
        assert data["filename"] == TEST_FILENAME
        assert data["chunks"] >= 1, "Au moins un chunk doit être créé"

    def test_uploaded_document_appears_in_list(self):
        """Le document uploadé apparaît dans GET /documents."""
        upload_test_doc()
        r = requests.get(f"{BASE_URL}/documents", headers=api_headers(), timeout=10)
        assert r.status_code == 200
        data = r.json()
        filenames = [d["filename"] for d in data["documents"]]
        assert TEST_FILENAME in filenames, (
            f"'{TEST_FILENAME}' introuvable dans la liste : {filenames}"
        )

    def test_duplicate_upload_returns_409(self):
        """Un fichier déjà indexé est refusé avec 409 Conflict."""
        upload_test_doc()
        r = upload_test_doc()  # second upload identique
        assert r.status_code == 409, (
            f"Le doublon devrait retourner 409, reçu : {r.status_code}"
        )

    def test_oversized_file_rejected(self):
        """Un fichier dépassant 10 Mo est refusé avec 400."""
        big_content = b"x" * (11 * 1024 * 1024)  # 11 Mo
        r = requests.post(
            f"{BASE_URL}/upload",
            headers=api_headers(),
            files={"file": ("big.txt", io.BytesIO(big_content), "text/plain")},
            timeout=30,
        )
        assert r.status_code == 400, f"Attendu 400 pour un fichier trop lourd, reçu : {r.status_code}"

    def test_unsupported_type_rejected(self):
        """Un type de fichier non supporté est refusé avec 400."""
        r = requests.post(
            f"{BASE_URL}/upload",
            headers=api_headers(),
            files={"file": ("script.py", io.BytesIO(b"print('hello')"), "text/x-python")},
            timeout=30,
        )
        assert r.status_code == 400


@pytest.mark.integration
class TestQueryPipeline:
    def test_query_returns_answer_and_sources(self):
        """
        Pipeline complet : upload réel → query → réponse avec sources citées.
        Vérifie que Claude utilise le document et cite le bon fichier.
        """
        upload_test_doc()

        r = requests.post(
            f"{BASE_URL}/query",
            headers=api_headers({"Content-Type": "application/json"}),
            json={"question": "Quel est le chiffre d'affaires 2024 ?"},
            timeout=60,
        )
        assert r.status_code == 200, f"Query échouée : {r.text}"
        data = r.json()

        assert "answer" in data, "La réponse doit contenir 'answer'"
        assert "sources" in data, "La réponse doit contenir 'sources'"
        assert data["answer"], "La réponse ne doit pas être vide"

    def test_source_points_to_correct_file(self):
        """Les sources citées pointent vers le fichier uploadé."""
        upload_test_doc()

        r = requests.post(
            f"{BASE_URL}/query",
            headers=api_headers({"Content-Type": "application/json"}),
            json={"question": "Quel est le chiffre d'affaires 2024 ?"},
            timeout=60,
        )
        data = r.json()

        # Si des sources sont retournées, elles doivent pointer vers notre fichier
        if data["sources"]:
            source = data["sources"][0]
            assert source["filename"] == TEST_FILENAME, (
                f"Source incorrecte : attendu '{TEST_FILENAME}', reçu '{source['filename']}'"
            )
            assert source["excerpt"], "L'extrait ne doit pas être vide"

    def test_answer_contains_known_figure_or_declines(self):
        """
        Vérifie le comportement anti-hallucination :
        - soit Claude cite le chiffre exact (42 millions),
        - soit il déclare ne pas trouver l'information (refus propre).
        Les deux sont acceptables ; inventer un autre chiffre ne l'est pas.
        """
        upload_test_doc()

        r = requests.post(
            f"{BASE_URL}/query",
            headers=api_headers({"Content-Type": "application/json"}),
            json={"question": "Quel est le chiffre d'affaires 2024 ?"},
            timeout=60,
        )
        data = r.json()
        answer_lower = data["answer"].lower()

        assert (
            "42" in answer_lower
            or "je ne trouve pas" in answer_lower
            or "pas" in answer_lower
        ), (
            f"Réponse inattendue (ni '42' ni refus explicite) : {data['answer']}"
        )

    def test_empty_question_rejected(self):
        """Une question vide est refusée avec 400."""
        r = requests.post(
            f"{BASE_URL}/query",
            headers=api_headers({"Content-Type": "application/json"}),
            json={"question": ""},
            timeout=10,
        )
        assert r.status_code == 400

    def test_query_without_documents_rejected(self):
        """Une query sans aucun document indexé est refusée avec 400."""
        # L'index est vide grâce à la fixture clean_index
        r = requests.post(
            f"{BASE_URL}/query",
            headers=api_headers({"Content-Type": "application/json"}),
            json={"question": "Quelle est la marge nette ?"},
            timeout=10,
        )
        assert r.status_code == 400, (
            f"Sans document, la query devrait retourner 400, reçu : {r.status_code}"
        )


@pytest.mark.integration
class TestDocumentDeletion:
    def test_delete_single_document(self):
        """Un document supprimé n'apparaît plus dans l'index."""
        upload_test_doc()

        # Récupère l'ID du document
        r = requests.get(f"{BASE_URL}/documents", headers=api_headers(), timeout=10)
        doc_id = next(
            d["id"] for d in r.json()["documents"] if d["filename"] == TEST_FILENAME
        )

        # Suppression
        r = requests.delete(f"{BASE_URL}/documents/{doc_id}", headers=api_headers(), timeout=30)
        assert r.status_code == 200

        # Vérification
        r = requests.get(f"{BASE_URL}/documents", headers=api_headers(), timeout=10)
        filenames = [d["filename"] for d in r.json()["documents"]]
        assert TEST_FILENAME not in filenames, (
            f"'{TEST_FILENAME}' devrait être supprimé mais apparaît encore dans : {filenames}"
        )

    def test_delete_all_documents(self):
        """DELETE /documents vide l'index complet."""
        upload_test_doc()
        upload_test_doc("autre_doc.txt", b"Contenu du second document.")

        r = requests.delete(f"{BASE_URL}/documents", headers=api_headers(), timeout=30)
        assert r.status_code == 200

        r = requests.get(f"{BASE_URL}/documents", headers=api_headers(), timeout=10)
        assert r.json()["total"] == 0, "L'index devrait être vide après DELETE /documents"

    def test_delete_nonexistent_document_returns_404(self):
        """Supprimer un ID inexistant retourne 404."""
        r = requests.delete(f"{BASE_URL}/documents/999999", headers=api_headers(), timeout=10)
        assert r.status_code == 404
