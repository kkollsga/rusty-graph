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

## Storage Modes & Optimization Priority

KGLite has three storage modes: `Default` (in-memory petgraph), `Mapped` (mmap-backed columns), and `Disk` (CSR + mmap). **In-memory is the core product.** Disk/mapped are addons for large-graph exploration (e.g., Wikidata).

**When optimizing, in-memory wins every time.** If an optimization helps both in-memory and disk, great. If there's a conflict (e.g., adding overhead to the hot path to protect against disk-scale explosions), find a disk-specific workaround instead. Never regress in-memory performance for disk safety.

- Small graphs (legal, code, domain): ~100K–500K nodes, all in-memory, full scans are fast (<10 ms). Don't add guardrails that penalize these.
- Large graphs (Wikidata): 100M+ nodes, disk-backed, full scans are catastrophic. Safeguards needed but must be gated behind storage mode or graph-size thresholds.
- Cypher query planner/executor is shared across all modes. Any changes to `pattern_matching.rs` or `cypher/executor.rs` affect everyone — benchmark on small graphs before merging.

## Performance Work Protocol

Before starting any performance-related code changes:
1. **Baseline first** — write a benchmark covering all code paths being touched. Run it, record numbers.
2. **Benchmark on in-memory graphs** — small/medium graphs (legal-scale) must not regress. This is the gate.
3. **Measure after** — re-run the same benchmark after changes. Report before/after.
4. **Disk benchmarks are secondary** — nice to show improvement, but never at the cost of in-memory.

## Key Patterns

- PyO3: `&self` for read-only, return `PyResult<Py<PyAny>>`, use `Python::attach()`.
- Use `.cast::<T>()` not `.downcast::<T>()` (deprecated in pyo3 0.27+).
- Private helpers go in non-`#[pymethods]` `impl KnowledgeGraph` block.
- Value conversion: `py_out::value_to_py()` and `py_out::nodeinfo_to_pydict()`.

## Storage-backend work (0.8.0 refactor)

Live architecture doc: `ARCHITECTURE.md` at repo root. Full refactor
plan: `todo.md` at repo root.

- **Reads go on `GraphRead`, mutations on `GraphWrite: GraphRead`.** Add new storage ops to the trait first, not as inherent `GraphBackend` methods. Both traits live in `src/graph/storage/mod.rs`.
- **Transactions stay on `DirGraph`.** OCC `version`, `read_only`, `schema_locked`, and validation helpers are not trait surface — see ARCHITECTURE.md.
- **No shims / no `#[deprecated]`.** Obsoleted code gets deleted in the same PR as its replacement. Clean break for 0.8.0.
- **`&impl GraphRead` / `&mut impl GraphWrite` in hot loops, `&dyn …` at boundaries.** Monomorphise tight scans; tolerate vtable cost only where API shape demands it.
- **Parity oracles**: `tests/test_storage_parity.py`, `tests/test_phase1_parity.py`, `tests/test_phase2_parity.py`. Gated behind `pytest -m parity`. Must stay green after any backend-touching change.

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

**NEVER push — the user pushes manually.** Before a release:
1. Confirm version number with user (bump patch +0.0.1)
2. Update `Cargo.toml` version + promote changelog
3. Commit, then let the user push

Version source of truth: `Cargo.toml` line 3.
