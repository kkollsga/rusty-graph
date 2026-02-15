#!/usr/bin/env python3
"""
Example: Building a knowledge graph from source code with kglite.code_tree.

Parses a codebase into a navigable knowledge graph with nodes for functions,
classes/structs, modules, enums, and files, plus edges for CALLS, DEFINES,
HAS_METHOD, HAS_SUBMODULE, IMPLEMENTS, EXTENDS, and IMPORTS relationships.

Supports Rust, Python, TypeScript, and JavaScript out of the box.

Requirements:
    pip install kglite[code-tree]

Usage:
    python build_code_graph.py                   # parse KGLite's own src/
    python build_code_graph.py /path/to/project  # parse any project
"""

import sys
from pathlib import Path

from kglite.code_tree import build

# ── 1. Build the graph ─────────────────────────────────────────────────

src_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "src"

print(f"Parsing: {src_dir}")
graph = build(src_dir, verbose=True)

# ── 2. Schema overview ─────────────────────────────────────────────────

print("\n=== Schema ===")
schema = graph.schema()
print(f"Nodes: {schema['node_count']}, Edges: {schema['edge_count']}")

for name, info in schema["node_types"].items():
    print(f"  :{name} ({info['count']} nodes)")
for name, info in schema["connection_types"].items():
    print(f"  -[:{name}]- ({info['count']} edges)")

# ── 3. Navigate the code with Cypher queries ───────────────────────────

# Find the most-called functions (hub functions in the codebase)
print("\n=== Most-Called Functions (top 10) ===")
for row in graph.cypher("""
    MATCH (caller:Function)-[:CALLS]->(f:Function)
    RETURN f.name AS function, f.file_path AS file, count(caller) AS callers
    ORDER BY callers DESC
    LIMIT 10
"""):
    print(f"  {row['function']} ({row['file']}): {row['callers']} callers")

# Largest files by number of definitions
print("\n=== Largest Files ===")
for row in graph.cypher("""
    MATCH (f:File)-[:DEFINES]->(item)
    RETURN f.filename AS file, f.loc AS lines, count(item) AS definitions
    ORDER BY definitions DESC
    LIMIT 10
"""):
    print(f"  {row['file']}: {row['definitions']} items, {row['lines']} lines")

# Module hierarchy
print("\n=== Module Tree ===")
for row in graph.cypher("""
    MATCH (parent:Module)-[:HAS_SUBMODULE]->(child:Module)
    RETURN parent.name AS parent, collect(child.name) AS children
    ORDER BY parent
"""):
    print(f"  {row['parent']} -> {', '.join(row['children'])}")

# Cross-module function calls (functions calling into a different file)
print("\n=== Cross-File Call Hotspots (top 10) ===")
for row in graph.cypher("""
    MATCH (f:Function)-[:CALLS]->(g:Function)
    WHERE f.file_path <> g.file_path
    RETURN f.name AS caller, f.file_path AS from_file,
           g.name AS callee, g.file_path AS to_file
    LIMIT 10
"""):
    print(f"  {row['caller']} ({row['from_file']}) -> {row['callee']} ({row['to_file']})")

# ── 4. Graph algorithms ────────────────────────────────────────────────

# PageRank on the call graph — finds the most "important" functions
print("\n=== PageRank: Call Graph (top 10) ===")
for row in graph.pagerank(connection_types=["CALLS"], top_k=10):
    print(f"  {row['title']} ({row['type']}): {row['score']:.4f}")

# Betweenness centrality — finds bridge functions between modules
print("\n=== Betweenness Centrality: Call Graph (top 10) ===")
for row in graph.betweenness_centrality(connection_types=["CALLS"], top_k=10):
    print(f"  {row['title']} ({row['type']}): {row['score']:.4f}")

# ── 5. Save and reload ─────────────────────────────────────────────────

output = "code_graph.kgl"
graph.save(output)
print(f"\nGraph saved to {output}")
print(f"Reload with: graph = kglite.load('{output}')")
