"""Custom-embedder factory loader for ``kglite-mcp-server``.

A manifest may declare ``embedder.module`` / ``embedder.class`` /
``embedder.kwargs`` to swap in a project-supplied embedder
(typically one with cooldown-based unload semantics, e.g.
``BAAI/bge-m3`` on consumer hardware). The factory is loaded the
same way as ``python:`` tool hooks — file paths resolved relative
to the manifest, importlib without ``sys.path`` mutation — and
gated by two signals: ``trust.allow_embedder: true`` in the yaml
plus ``--trust-tools`` on the CLI.

The CLI's ``--embedder MODEL_NAME`` shortcut still works for the
common case of "wrap a sentence-transformers model"; the factory
path is the escape hatch for everything else.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from kglite.mcp_server.manifest import EmbedderConfig, ManifestError


def load_embedder(
    config: EmbedderConfig,
    *,
    manifest_dir: Path,
    trust_flag: bool,
    allow_embedder: bool,
):
    """Resolve the manifest-declared embedder and return an instance.

    Trust signals must both be set; either alone refuses to load.
    Raises :class:`ManifestError` on any failure.
    """
    if not allow_embedder:
        raise ManifestError("manifest declares an embedder but `trust.allow_embedder: true` is not set")
    if not trust_flag:
        raise ManifestError(
            "manifest declares an embedder but the CLI was started without --trust-tools "
            "(refusing to load arbitrary code)"
        )

    module_path = (manifest_dir / config.module).resolve()
    if not module_path.is_file():
        raise ManifestError(f"embedder module {config.module!r} resolves to {str(module_path)!r} which does not exist")

    module_name = f"_kglite_mcp_user_embedder_{config.klass}"
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise ManifestError(f"embedder: could not load module from {module_path}")

    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as e:
        raise ManifestError(f"embedder: error executing {module_path}: {e}") from e

    klass = getattr(module, config.klass, None)
    if klass is None:
        raise ManifestError(f"embedder: class {config.klass!r} not found in {module_path}")
    if not callable(klass):
        raise ManifestError(f"embedder: {config.klass!r} in {module_path} is not callable")

    try:
        return klass(**config.kwargs)
    except Exception as e:
        raise ManifestError(f"embedder: failed to instantiate {config.klass}: {e}") from e


def build_sentence_transformer_embedder(model_name: str):
    """Build a lightweight wrapper around ``sentence-transformers``.

    This is the fallback used by ``--embedder MODEL_NAME``. The model
    stays loaded for the lifetime of the server (``load`` / ``unload``
    are no-ops), which is fine for small models but a poor fit for
    larger ones — use the manifest-declared embedder factory in that
    case.
    """
    from sentence_transformers import SentenceTransformer

    class _Embedder:
        def __init__(self, name: str) -> None:
            self._model = SentenceTransformer(name)
            self.dimension = self._model.get_sentence_embedding_dimension()

        def embed(self, texts: list[str]) -> list[list[float]]:
            return self._model.encode(texts, show_progress_bar=False).tolist()

        def load(self) -> None:
            pass

        def unload(self) -> None:
            pass

    return _Embedder(model_name)
