# Code Tree

Parse multi-language codebases into KGLite knowledge graphs using [tree-sitter](https://tree-sitter.github.io/tree-sitter/). Extracts functions, classes/structs, enums, traits/interfaces, modules, and their relationships.

```bash
pip install kglite[code-tree]
```

## Quick Start

```python
from kglite.code_tree import build

graph = build(".")  # auto-detects pyproject.toml / Cargo.toml

# What are the most-called functions?
graph.cypher("""
    MATCH (caller:Function)-[:CALLS]->(f:Function)
    RETURN f.name AS function, count(caller) AS callers
    ORDER BY callers DESC LIMIT 10
""")

# Label-optional matching — search across all node types
graph.cypher("""
    MATCH (n {name: 'execute'})
    RETURN n.type, n.name, n.file_path, n.line_number
""")

# Save for later
graph.save("codebase.kgl")
```

## Code Exploration Methods

```python
# Find entities by name (searches all code entity types)
graph.find("execute")
graph.find("KnowledgeGraph", node_type="Struct")
graph.find("exec", match_type="contains")       # case-insensitive substring
graph.find("Knowl", match_type="starts_with")    # case-insensitive prefix

# Get source location — single or batch
graph.source("execute_single_clause")
# {'file_path': 'src/graph/cypher/executor.rs', 'line_number': 165,
#  'end_line': 205, 'line_count': 41, 'signature': '...'}
graph.source(["KnowledgeGraph", "build", "execute"])

# Get full neighborhood of an entity
graph.context("KnowledgeGraph")
# {'node': {...}, 'defined_in': 'src/graph/mod.rs',
#  'HAS_METHOD': [...], 'IMPLEMENTS': [...], 'called_by': [...]}

# File table of contents — all entities defined in a file
graph.toc("src/graph/mod.rs")
# {'file': '...', 'entities': [...], 'summary': {'Function': 4, 'Struct': 2}}
```

## Supported Languages

| Language | Extensions |
|----------|------------|
| Rust | `.rs` |
| Python | `.py`, `.pyi` |
| TypeScript | `.ts`, `.tsx` |
| JavaScript | `.js`, `.jsx`, `.mjs` |
| Go | `.go` |
| Java | `.java` |
| C# | `.cs` |
| C | `.c`, `.h` |
| C++ | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh`, `.hxx` |

## Graph Schema

**Node types:** `Project`, `Dependency`, `File`, `Module`, `Function`, `Struct`, `Class`, `Enum`, `Trait`, `Protocol`, `Interface`, `Attribute`, `Constant`

**Relationship types:** `DEPENDS_ON` (Project→Dependency), `HAS_SOURCE` (Project→File), `DEFINES` (File→item), `CALLS` (Function→Function), `HAS_METHOD` (Struct/Class→Function), `HAS_ATTRIBUTE` (Struct/Class→Attribute), `HAS_SUBMODULE` (Module→Module), `IMPLEMENTS` (type→trait), `EXTENDS` (class→class), `IMPORTS` (File→Module), `USES_TYPE`, `EXPOSES` (Module→item)

## Options

```python
graph = build(".")                           # auto-detect manifest (pyproject.toml, Cargo.toml)
graph = build("pyproject.toml")              # explicit manifest file
graph = build("/path/to/src")                # directory scan (fallback when no manifest)
graph = build(".", include_tests=True)       # include test directories
graph = build(".", save_to="code.kgl", verbose=True)
```

When a manifest is detected, `build()` reads project metadata (name, version, dependencies) and only scans declared source directories — avoiding `.venv/`, `target/`, `node_modules/`, etc.
