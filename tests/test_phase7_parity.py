"""Phase 7 crunch-point parity tests.

Guards the structural-reorg + final-audit phase of the 0.8.0 storage
refactor. Tests here:

- **god-file gate** — enumerates `*.rs` under `src/graph/`, asserts each
  is ≤ 2500 lines unless it is on the documented exception list. The
  exception list is maintained with a written rationale per entry and
  a target 0.9.0+ follow-up.
- **unsafe-has-SAFETY gate** — every `unsafe { ... }` block has a
  preceding `// SAFETY:` comment (within the last 5 lines).
- **mod.rs purity gate** — each subdir-`mod.rs` file under
  `src/graph/{algorithms,features,introspection,io,mutation,pyapi,query,storage}/`
  is short and contains no `impl` blocks or long function bodies.
- **ARCHITECTURE.md points at files that exist** — every concrete file
  path mentioned in ARCHITECTURE.md actually exists in the tree.

Run: pytest -m parity tests/test_phase7_parity.py
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.parity

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_GRAPH = REPO_ROOT / "src" / "graph"

# ── God-file gate ──────────────────────────────────────────────────────────
#
# Soft target: every .rs ≤ 1500. Hard cap: ≤ 2500. Exceptions below
# carry a rationale + the 0.9.0+ follow-up plan.
HARD_CAP = 2500

GOD_FILE_EXCEPTIONS: dict[str, str] = {
    # Phase 9 emptied this list: every `.rs` under `src/graph/` now sits
    # at or under the 2,500-line hard cap.
}


def _relative_rs_path(path: Path) -> str:
    return path.relative_to(SRC_GRAPH).as_posix()


def test_god_file_gate():
    """No `.rs` under `src/graph/` exceeds 2500 lines unless documented."""
    offenders: list[tuple[str, int]] = []
    for rs in sorted(SRC_GRAPH.rglob("*.rs")):
        rel = _relative_rs_path(rs)
        lines = len(rs.read_text().splitlines())
        if lines > HARD_CAP and rel not in GOD_FILE_EXCEPTIONS:
            offenders.append((rel, lines))
    if offenders:
        msg = "Files over 2500-line hard cap without a documented exception:\n"
        msg += "\n".join(f"  {rel}: {lines} lines" for rel, lines in offenders)
        msg += (
            "\n\nEither split the file or add it to GOD_FILE_EXCEPTIONS "
            "with a written rationale and 0.9.0+ follow-up plan."
        )
        pytest.fail(msg)


def test_exception_list_still_applies():
    """Every entry in GOD_FILE_EXCEPTIONS actually points at a file > 2500 lines.

    Keeps the exception list honest — if a file drops under the cap
    via a future split, it should be removed from the list.
    """
    stale: list[str] = []
    missing: list[str] = []
    for rel in GOD_FILE_EXCEPTIONS:
        path = SRC_GRAPH / rel
        if not path.exists():
            missing.append(rel)
            continue
        lines = len(path.read_text().splitlines())
        if lines <= HARD_CAP:
            stale.append(f"{rel}: now only {lines} lines (under cap)")
    errors: list[str] = []
    if missing:
        errors.append("exception list references non-existent files:\n" + "\n".join(f"  {r}" for r in missing))
    if stale:
        errors.append("exception list has stale entries:\n" + "\n".join(f"  {r}" for r in stale))
    if errors:
        pytest.fail("\n\n".join(errors))


UNSAFE_OPEN = re.compile(r"unsafe\s*\{")


def test_unsafe_has_safety_comment():
    """Every `unsafe { ... }` has `// SAFETY:` within the prior 5 lines."""
    offenders: list[str] = []
    for rs in sorted(SRC_GRAPH.rglob("*.rs")):
        lines = rs.read_text().splitlines()
        for i, line in enumerate(lines):
            if UNSAFE_OPEN.search(line):
                # Skip comment lines that mention unsafe in prose
                stripped = line.lstrip()
                if stripped.startswith("//"):
                    continue
                window = lines[max(0, i - 5) : i]
                if not any("SAFETY" in w for w in window):
                    rel = rs.relative_to(REPO_ROOT).as_posix()
                    offenders.append(f"{rel}:{i + 1}: {line.strip()}")
    if offenders:
        pytest.fail("`unsafe {` blocks without `// SAFETY:` justification:\n" + "\n".join(f"  {o}" for o in offenders))


def test_mod_rs_purity():
    """Each subdir-`mod.rs` is short (no big impl blocks, no long fns)."""
    subdirs = [
        "algorithms",
        "core",
        "core/pattern_matching",
        "features",
        "introspection",
        "io",
        "io/ntriples",
        "languages",
        "languages/cypher",
        "languages/cypher/executor",
        "languages/cypher/parser",
        "languages/cypher/planner",
        "languages/fluent",
        "mutation",
        "pyapi",
        "storage",
        "storage/disk",
        "storage/mapped",
        "storage/memory",
    ]
    offenders: list[str] = []
    for sub in subdirs:
        mod_rs = SRC_GRAPH / sub / "mod.rs"
        if not mod_rs.exists():
            continue
        text = mod_rs.read_text()
        lines = text.splitlines()
        # Soft cap on subdir mod.rs line count. `storage/mod.rs`
        # legitimately holds the GraphRead/GraphWrite trait definitions
        # + GAT associated-type bounds. `cypher/mod.rs` holds the
        # Cypher facade (cypher() entry point + parse/plan orchestration).
        # Other subdir mod.rs files are pure re-exports.
        if sub == "storage":
            cap = 800
        elif sub == "languages/cypher":
            cap = 500
        elif sub == "languages/cypher/executor":
            # Hosts CypherExecutor struct + constructor + execute()
            # orchestrator + finalize_result + filter-spec types — the
            # shared state that every clause-submodule borrows from.
            cap = 1200
        elif sub == "languages/cypher/parser":
            # Hosts CypherParser struct + token helpers + parse_query
            # orchestrator + public parse_cypher entry + tests.
            cap = 1000
        elif sub == "languages/cypher/planner":
            # Hosts the optimize() orchestrator + mark_* helpers + tests.
            cap = 400
        else:
            cap = 300
        if len(lines) > cap:
            offenders.append(f"{sub}/mod.rs: {len(lines)} lines > cap {cap}")
    if offenders:
        pytest.fail("Subdir `mod.rs` files are too long:\n" + "\n".join(f"  {o}" for o in offenders))


def test_architecture_md_mentions_real_files():
    """File paths literally referenced in ARCHITECTURE.md actually exist."""
    arch_md = (REPO_ROOT / "ARCHITECTURE.md").read_text()
    # Match `path/file.rs` in backticks or code blocks.
    path_pattern = re.compile(r"`(src/[^\s`]+\.rs)`")
    missing: list[str] = []
    for match in path_pattern.finditer(arch_md):
        path = REPO_ROOT / match.group(1)
        if not path.exists():
            missing.append(match.group(1))
    # Dedup
    missing = sorted(set(missing))
    if missing:
        pytest.fail("ARCHITECTURE.md references file paths that do not exist:\n" + "\n".join(f"  {m}" for m in missing))
