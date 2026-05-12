"""`extensions.cypher_preprocessor` builder — manifest-declared hook.

Loads a Python `.py` module specified by the manifest, instantiates the
declared class (or resolves the declared free function), and wraps it
in a tiny dispatcher. The dispatcher's `rewrite(query, params)` is
called on every `cypher_query` / `tools[].cypher` invocation before
the query reaches `graph.cypher(...)`.

Trust gate: requires `trust.allow_query_preprocessor: true` in the
manifest. Mirrors how `extensions.embedder` requires
`trust.allow_embedder`.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import sys
from typing import Any, Callable

log = logging.getLogger("kglite.mcp_server.preprocessor")


class PreprocessorError(ValueError):
    """Raised when manifest-declared preprocessor loading fails. Message
    is operator-facing — surfaces to stderr at boot."""


@dataclass
class Preprocessor:
    """Wraps the user-supplied callable. The wrapper layer normalises
    whether the manifest declared a `class:` (instantiated with kwargs)
    or a `function:` (module-level free function), so the caller can
    always invoke `.rewrite(query, params)`."""

    rewrite: Callable[[str, dict | None], tuple[str, dict | None]]
    module_path: Path


def from_manifest_value(
    value: Any,
    base_dir: Path,
    trust_allowed: bool,
) -> Preprocessor | None:
    """Parse `extensions.cypher_preprocessor` from the manifest. Returns
    None when the block is absent. Raises `PreprocessorError` on
    misconfiguration — the caller surfaces the message to stderr."""
    if value is None:
        return None
    if not trust_allowed:
        raise PreprocessorError("extensions.cypher_preprocessor requires trust.allow_query_preprocessor: true")
    if not isinstance(value, dict):
        raise PreprocessorError(f"extensions.cypher_preprocessor must be a mapping (got {value!r})")

    allowed_keys = {"module", "class", "function", "kwargs"}
    unknown = set(value.keys()) - allowed_keys
    if unknown:
        raise PreprocessorError(
            f"extensions.cypher_preprocessor: unknown keys {sorted(unknown)}. Allowed: {sorted(allowed_keys)}"
        )

    module_raw = value.get("module")
    if not isinstance(module_raw, str) or not module_raw:
        raise PreprocessorError("extensions.cypher_preprocessor.module must be a non-empty string")

    class_name = value.get("class")
    function_name = value.get("function")
    if class_name is None and function_name is None:
        raise PreprocessorError("extensions.cypher_preprocessor must declare exactly one of `class:` or `function:`")
    if class_name is not None and function_name is not None:
        raise PreprocessorError("extensions.cypher_preprocessor cannot declare both `class:` and `function:`")

    kwargs = value.get("kwargs", {}) or {}
    if not isinstance(kwargs, dict):
        raise PreprocessorError("extensions.cypher_preprocessor.kwargs must be a mapping")

    # Resolve module path against the manifest dir (matches the
    # documented rule: manifest-relative paths everywhere).
    module_path = (base_dir / module_raw).resolve() if not Path(module_raw).is_absolute() else Path(module_raw)
    if not module_path.is_file():
        raise PreprocessorError(f"extensions.cypher_preprocessor.module file does not exist: {module_path}")

    module = _load_module(module_path)

    if class_name is not None:
        if not isinstance(class_name, str):
            raise PreprocessorError("extensions.cypher_preprocessor.class must be a string")
        if not hasattr(module, class_name):
            raise PreprocessorError(f"extensions.cypher_preprocessor: class {class_name!r} not found in {module_path}")
        cls = getattr(module, class_name)
        try:
            instance = cls(**kwargs)
        except TypeError as e:
            raise PreprocessorError(f"extensions.cypher_preprocessor: failed to construct {class_name}: {e}") from None
        if not callable(getattr(instance, "rewrite", None)):
            raise PreprocessorError(f"extensions.cypher_preprocessor: {class_name} instance has no callable rewrite()")
        rewrite = instance.rewrite
    else:
        if not isinstance(function_name, str):
            raise PreprocessorError("extensions.cypher_preprocessor.function must be a string")
        if not hasattr(module, function_name):
            raise PreprocessorError(
                f"extensions.cypher_preprocessor: function {function_name!r} not found in {module_path}"
            )
        fn = getattr(module, function_name)
        if not callable(fn):
            raise PreprocessorError(f"extensions.cypher_preprocessor: {function_name} is not callable")
        rewrite = fn

    log.info(
        "registered cypher_preprocessor (module=%s, %s=%s)",
        module_path,
        "class" if class_name else "function",
        class_name or function_name,
    )
    return Preprocessor(rewrite=rewrite, module_path=module_path)


def _load_module(path: Path):
    """Import a .py file by path with a synthetic module name. Uses
    importlib.util so the path doesn't need to be on sys.path."""
    # Synthetic name keeps the module unique per path even if multiple
    # preprocessors live under different manifests.
    synthetic_name = f"_kglite_cypher_preprocessor_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(synthetic_name, str(path))
    if spec is None or spec.loader is None:
        raise PreprocessorError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[synthetic_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001 — surface as boot error
        sys.modules.pop(synthetic_name, None)
        raise PreprocessorError(f"failed to import {path}: {e}") from None
    return module


def apply(
    preprocessor: Preprocessor | None,
    query: str,
    params: dict | None,
) -> tuple[str, dict | None] | str:
    """Run the preprocessor (if configured) and return the new
    (query, params). On exception, returns the user-facing error
    string `preprocessor: <message>` instead — caller treats a str
    return as a short-circuit and forwards it as the tool response
    body. Mirrors the embedder factory's "load-time errors raise,
    runtime errors surface as text" split."""
    if preprocessor is None:
        return query, params
    try:
        new_query, new_params = preprocessor.rewrite(query, params)
    except (ValueError, TypeError) as e:
        # Operator-blessed shape: agent sees the message verbatim,
        # no traceback. Per the operator's spec
        # (test_cypher_preprocessor_exception_returns_clean_error).
        return f"preprocessor: {e}"
    except Exception as e:  # noqa: BLE001 — unexpected, but never trace
        log.warning("cypher_preprocessor raised unexpected %s: %s", type(e).__name__, e)
        return f"preprocessor: {type(e).__name__}: {e}"
    if not isinstance(new_query, str):
        return f"preprocessor: rewrite() must return (str, dict|None); got query={type(new_query).__name__}"
    if new_params is not None and not isinstance(new_params, dict):
        return f"preprocessor: rewrite() must return (str, dict|None); got params={type(new_params).__name__}"
    return new_query, new_params
