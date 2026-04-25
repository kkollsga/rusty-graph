"""KGLite dataset helpers — opinionated builders for well-known public
datasets (Wikidata RDF dumps, etc.). Each submodule wraps the
fetch + maintenance + build cycle behind a single entry point so
applications can treat a public dataset as a typed Python value.

Submodules:
    wikidata - Wikimedia Foundation's `latest-truthy` RDF dumps.
"""

from . import sodir, wikidata

__all__ = ["wikidata", "sodir"]
