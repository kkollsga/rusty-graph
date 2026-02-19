# KGLite — Claude Code Conventions

## Build & Test

```bash
# Always activate the venv and unset CONDA_PREFIX before building
source .venv/bin/activate && unset CONDA_PREFIX

# Build the Rust extension into the venv
maturin develop              # or: make dev

# Run tests
make test                    # Rust + Python
make test-rust               # Rust unit tests only
make test-py                 # Python tests only
cargo check                  # fast compile check (no codegen)
cargo fmt                    # format Rust code
cargo clippy -- -D warnings  # lint
make lint                    # fmt --check + clippy (run before pushing)
```

**Before pushing:** Always run `make lint` to catch formatting and clippy issues that CI will reject.

## Architecture

- **Rust core** (`src/`): `KnowledgeGraph` struct with `#[pymethods]` via PyO3. Graph storage uses `petgraph` with `DirGraph` wrapper in `schema.rs`.
- **Python package** (`kglite/`): thin wrapper + `code_tree/` subpackage for tree-sitter based codebase parsing.
- **Type stubs** (`kglite/__init__.pyi`): must be updated when adding or changing any `#[pymethods]` function.
- **Cypher engine** (`src/graph/cypher/`): parser → AST → executor pipeline. Supports MATCH, WHERE, RETURN, CREATE, SET, DELETE, aggregations, path patterns.
- **Code tree parser** (`kglite/code_tree/`): language-specific parsers in `parsers/`, graph builder in `builder.py`. Outputs a `KnowledgeGraph` with code entities.
- **MCP server** (`examples/mcp_server.py`): FastMCP server exposing the graph to AI agents.
- **Introspection** (`src/graph/introspection.rs`): `agent_describe()` generates XML schema description for AI agent consumption.

## Key Patterns

- **PyO3 methods**: `&self` for read-only, return `PyResult<Py<PyAny>>`, use `Python::attach()` for GIL access.
- **NodeData fields**: `id` (qualified_name for code entities), `title` (display name), `node_type`, `properties` HashMap.
- **Type indices**: `HashMap<String, Vec<NodeIndex>>` for fast node lookup by type.
- **Private helpers**: put shared logic in a non-`#[pymethods]` `impl KnowledgeGraph` block (around line 140-310 in mod.rs).
- **Value conversion**: use `py_out::value_to_py()` and `py_out::nodeinfo_to_pydict()` for Rust→Python.
- **pyo3 0.27+**: use `.cast::<T>()` not `.downcast::<T>()` (deprecated). Use `.into()` for `Bound<PyDict>` → `Py<PyAny>` in non-pymethods blocks.

## Changelog Rule

**Always update `CHANGELOG.md`** when making user-visible changes. Add entries to the `[Unreleased]` section under the appropriate category (Added, Changed, Fixed, Removed).

Skip changelog updates for: internal refactors, CI changes, test-only changes, formatting fixes.

## Commit Messages

Format: `type: short description`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

## Files to Update Together

When adding a new `#[pymethods]` function:
1. `src/graph/mod.rs` — the implementation
2. `src/graph/introspection.rs` — add to `agent_describe()` API section + notes
3. `kglite/__init__.pyi` — type stub
4. `examples/mcp_server.py` — MCP tool (if agent-facing)
5. `CHANGELOG.md` — `[Unreleased]` → Added
6. `README.md` — if it's a major feature

## Version

Single source of truth: `Cargo.toml` line 3. `pyproject.toml` reads it dynamically via maturin.
