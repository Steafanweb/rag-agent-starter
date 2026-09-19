import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# Le header apparaît dans Swagger (icône cadenas → "Authorize")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    """
    Vérifie le header X-API-Key sur chaque endpoint protégé.

    - Si API_KEY n'est pas défini dans l'environnement → auth désactivée (mode dev local).
    - En production, toujours définir API_KEY dans .env.
    """
    expected = os.getenv("API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=403, detail="Clé API invalide ou absente.")
