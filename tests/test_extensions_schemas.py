"""Regression tests for the published `extensions:` JSON Schemas.

0.9.25: schemas live under `docs/schemas/extensions/` and document
the validated `extensions.*` blocks (csv_http_server, embedder,
cypher_preprocessor). The contract is two-way:

1. Anything the Python parsers accept must validate against the
   schema. If a parser starts accepting a new shape without a schema
   update, agents and operators reading the docs see stale info —
   we want a loud test failure instead.

2. Anything the schema rejects, the Python parser should also reject
   at boot (and vice versa, where the parser raises). The schema is
   the agent-facing source of truth for "what shapes are legal."

These tests anchor both directions so future drift fails loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Tests use jsonschema to validate the published Draft 2020-12 schemas.
# The `[mcp]` extras pull jsonschema for this; in stripped envs (no
# extras installed) the tests skip cleanly rather than ImportError.
jsonschema = pytest.importorskip("jsonschema")

SCHEMAS_DIR = Path(__file__).parent.parent / "docs" / "schemas" / "extensions"


def _load(name: str) -> dict:
    path = SCHEMAS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# csv_http_server
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        {},
        {"port": 8765},
        {"port": 9000, "dir": "exports/"},
        {"port": 0, "dir": "/abs/path", "cors_origin": "https://app.example"},
        {"port": 65535, "cors_origin": None},
    ],
)
def test_csv_http_server_schema_accepts_valid_shapes(value):
    """Every shape the Python parser accepts validates."""
    jsonschema.validate(value, _load("csv_http_server"))


@pytest.mark.parametrize(
    "value",
    [
        {"port": -1},
        {"port": 65536},
        {"port": "8765"},
        {"port": 8765, "dir": 123},
        {"unknown_key": True},
        "not a mapping or bool",
        ["array"],
        42,
    ],
)
def test_csv_http_server_schema_rejects_invalid_shapes(value):
    """Shapes the Python parser rejects also fail schema validation."""
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, _load("csv_http_server"))


def test_csv_http_server_schema_matches_parser_acceptance():
    """Cross-check: every accepting case from the schema also rounds
    through `csv_http.from_manifest_value` without raising. Catches
    drift between the schema and the Python parser."""
    from kglite.mcp_server.csv_http import from_manifest_value

    base_dir = Path("/tmp")
    for value in [
        True,
        {"port": 8765},
        {"port": 9000, "dir": "exports/"},
        {"port": 0, "cors_origin": "*"},
        {},
    ]:
        # Should not raise.
        from_manifest_value(value, base_dir)


def test_csv_http_server_schema_matches_parser_rejection():
    """And the Python parser also rejects what the schema rejects."""
    from kglite.mcp_server.csv_http import from_manifest_value

    base_dir = Path("/tmp")
    for value in [
        {"port": -1},
        {"port": 65536},
        {"port": "8765"},
        "not a mapping or bool",
    ]:
        with pytest.raises(Exception):  # noqa: B017, BLE001 — parser raises ValueError
            from_manifest_value(value, base_dir)


# ---------------------------------------------------------------------------
# embedder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        {"backend": "fastembed", "model": "BAAI/bge-m3"},
        {"backend": "fastembed", "model": "bge-small-en-v1.5"},
        {"backend": "fastembed", "model": "BAAI/bge-m3", "cooldown": 1200},
        {"backend": "fastembed", "model": "BAAI/bge-m3", "cooldown": 0},
        {"backend": "fastembed", "model": "all-MiniLM-L6-v2"},
        {"backend": "fastembed", "model": "intfloat/multilingual-e5-large"},
    ],
)
def test_embedder_schema_accepts_valid_shapes(value):
    jsonschema.validate(value, _load("embedder"))


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"backend": "fastembed"},
        {"model": "BAAI/bge-m3"},
        {"backend": "other", "model": "BAAI/bge-m3"},
        {"backend": "fastembed", "model": "unknown-model"},
        {"backend": "fastembed", "model": "BAAI/bge-m3", "cooldown": -1},
        {"backend": "fastembed", "model": "BAAI/bge-m3", "cooldown": "long"},
        {"backend": "fastembed", "model": "BAAI/bge-m3", "extra_key": True},
    ],
)
def test_embedder_schema_rejects_invalid_shapes(value):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, _load("embedder"))


def test_embedder_schema_known_models_match_parser_catalog():
    """The schema's `model` enum must equal the parser's KNOWN_MODELS
    keys union BgeM3Embedder's "BAAI/bge-m3". Anything else means the
    docs are now lying."""
    from kglite.mcp_server.embedder import KNOWN_MODELS

    schema = _load("embedder")
    schema_models = set(schema["properties"]["model"]["enum"])
    parser_models = set(KNOWN_MODELS.keys())
    assert schema_models == parser_models, (
        f"schema/embedder.json's model enum drifted from "
        f"kglite.mcp_server.embedder.KNOWN_MODELS.\n"
        f"  in schema only: {sorted(schema_models - parser_models)}\n"
        f"  in parser only: {sorted(parser_models - schema_models)}"
    )


# ---------------------------------------------------------------------------
# cypher_preprocessor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        {"module": "./rewrites.py", "class": "Rewriter"},
        {"module": "/abs/path/p.py", "function": "rewrite"},
        {
            "module": "./p.py",
            "class": "P",
            "kwargs": {"log": True, "limit": 10},
        },
    ],
)
def test_cypher_preprocessor_schema_accepts_valid_shapes(value):
    jsonschema.validate(value, _load("cypher_preprocessor"))


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"module": "./p.py"},  # neither class nor function
        {
            "module": "./p.py",
            "class": "P",
            "function": "rewrite",  # both — mutually exclusive
        },
        {"module": 123, "class": "P"},
        {"module": "./p.py", "class": "1Bad"},  # invalid Python identifier
        {"module": "./p.py", "function": "with space"},
        {"module": "./p.py", "class": "P", "unknown_field": True},
        {"module": "./p.py", "class": "P", "kwargs": "not a mapping"},
    ],
)
def test_cypher_preprocessor_schema_rejects_invalid_shapes(value):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, _load("cypher_preprocessor"))
