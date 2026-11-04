"""OMEGA Utils — Interface Web (Flask).

Painel visual amigável sobre os módulos da biblioteca OMEGA Utils.
Execute a partir da raiz do projeto:

    python interface/app.py
"""

from interface.app import app, criar_app

__all__ = ["app", "criar_app"]
