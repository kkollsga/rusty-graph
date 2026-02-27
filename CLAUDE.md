# KGLite — Claude Code Conventions

## Build & Test

```bash
source .venv/bin/activate && unset CONDA_PREFIX
maturin develop              # build Rust extension into venv
make test                    # Rust + Python tests
make lint                    # fmt --check + clippy (always run before pushing)
```

## Architecture

- **Rust core** (`src/`): `KnowledgeGraph` with `#[pymethods]` via PyO3, `petgraph` storage.
- **Cypher engine** (`src/graph/cypher/`): parser → AST → executor.
- **Python package** (`kglite/`): thin wrapper + `code_tree/` (tree-sitter codebase parsing).
- **Type stubs** (`kglite/__init__.pyi`): source of truth for API docs — update when changing `#[pymethods]`.
- **Introspection** (`src/graph/introspection.rs`): `describe()` XML schema for AI agents.

## Key Patterns

- PyO3: `&self` for read-only, return `PyResult<Py<PyAny>>`, use `Python::attach()`.
- Use `.cast::<T>()` not `.downcast::<T>()` (deprecated in pyo3 0.27+).
- Private helpers go in non-`#[pymethods]` `impl KnowledgeGraph` block.
- Value conversion: `py_out::value_to_py()` and `py_out::nodeinfo_to_pydict()`.

## When Changing a `#[pymethods]` Function

1. `src/graph/mod.rs` — implementation
2. `kglite/__init__.pyi` — type stub + docstring
3. `src/graph/introspection.rs` — `describe()` output (if agent-facing)
4. `examples/mcp_server.py` — MCP tool (if agent-facing)
5. `CHANGELOG.md` — `[Unreleased]` section

## Documentation

Docs auto-rebuild at [kglite.readthedocs.io](https://kglite.readthedocs.io) on every push to `main`.

- **API reference**: auto-generated from docstrings in `kglite/__init__.pyi`
- **Cypher reference**: edit `CYPHER.md`
- **Fluent API reference**: edit `FLUENT.md`
- **Guide content**: edit `docs/guides/*.md`
- **README.md**: landing page only — do not duplicate guide content here

## Commits & Releases

Commit format: `type: short description` (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`)

Update `CHANGELOG.md` `[Unreleased]` for user-visible changes. Skip for internal refactors, CI, test-only, formatting.

**NEVER push without explicit user approval.** Before pushing:
1. Confirm version number with user (bump patch +0.0.1)
2. Update `Cargo.toml` version + promote changelog
3. Commit, then push after user approves

Version source of truth: `Cargo.toml` line 3.
