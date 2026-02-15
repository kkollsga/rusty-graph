#!/usr/bin/env python3
"""
Build a knowledge graph of a Rust codebase using tree-sitter + KGLite.

Parses Rust source files, extracts code entities (functions, structs, enums,
traits, modules), and loads them into a KGLite knowledge graph with
relationships (DEFINES, CALLS, IMPLEMENTS, HAS_METHOD, CONTAINS, IMPORTS).

Usage:
    python build_code_graph.py [src_directory]

Defaults to parsing KGLite's own src/ directory.

Dependencies:
    pip install tree-sitter tree-sitter-rust kglite pandas
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
import kglite
import tree_sitter_rust as ts_rust
from tree_sitter import Language, Parser

RUST_LANGUAGE = Language(ts_rust.language())

# ── Data classes for parsed entities ────────────────────────────────────


@dataclass
class FileInfo:
    path: str  # relative to src_root
    filename: str
    loc: int
    module_path: str  # e.g. "crate::graph::schema"
    mod_declarations: list[str] = field(default_factory=list)
    use_imports: list[str] = field(default_factory=list)


@dataclass
class FunctionInfo:
    name: str
    qualified_name: str
    visibility: str
    is_async: bool
    is_method: bool
    signature: str
    file_path: str
    line_number: int
    docstring: str | None
    return_type: str | None
    is_pymethod: bool
    calls: list[str] = field(default_factory=list)


@dataclass
class StructInfo:
    name: str
    qualified_name: str
    visibility: str
    file_path: str
    line_number: int
    docstring: str | None
    is_pyclass: bool


@dataclass
class EnumInfo:
    name: str
    qualified_name: str
    visibility: str
    file_path: str
    line_number: int
    docstring: str | None


@dataclass
class TraitInfo:
    name: str
    qualified_name: str
    visibility: str
    file_path: str
    line_number: int
    docstring: str | None


@dataclass
class ImplBlock:
    self_type: str
    trait_name: str | None  # None for inherent impl
    is_pymethods: bool
    methods: list[FunctionInfo] = field(default_factory=list)


# ── Helpers ─────────────────────────────────────────────────────────────


def node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf8")


def get_visibility(node) -> str:
    for child in node.children:
        if child.type == "visibility_modifier":
            text = child.text.decode("utf8")
            if "crate" in text:
                return "pub(crate)"
            return "pub"
    return "private"


def get_doc_comment(node, source: bytes) -> str | None:
    """Walk backward through siblings to collect /// doc comments."""
    doc_lines = []
    sibling = node.prev_named_sibling
    while sibling is not None:
        if sibling.type == "line_comment":
            text = node_text(sibling, source).strip()
            if text.startswith("///"):
                content = text[3:]
                if content.startswith(" "):
                    content = content[1:]
                doc_lines.insert(0, content)
                sibling = sibling.prev_named_sibling
                continue
        elif sibling.type == "attribute_item":
            sibling = sibling.prev_named_sibling
            continue
        break
    return "\n".join(doc_lines) if doc_lines else None


def get_attributes(node, source: bytes) -> list[str]:
    """Walk backward through siblings to collect #[...] attributes."""
    attrs = []
    sibling = node.prev_named_sibling
    while sibling is not None:
        if sibling.type == "attribute_item":
            attrs.insert(0, node_text(sibling, source))
            sibling = sibling.prev_named_sibling
            continue
        elif sibling.type == "line_comment":
            sibling = sibling.prev_named_sibling
            continue
        break
    return attrs


def has_pyclass(attrs: list[str]) -> bool:
    return any("#[pyclass" in a for a in attrs)


def is_pymethods_block(attrs: list[str]) -> bool:
    return any("#[pymethods]" in a for a in attrs)


def is_pymethod_fn(fn_attrs: list[str], impl_is_pymethods: bool) -> bool:
    if impl_is_pymethods:
        return True
    return any(m in a for a in fn_attrs for m in ("#[pyfunction]", "#[new]"))


def get_return_type(node, source: bytes) -> str | None:
    """Extract return type from a function_item node."""
    saw_arrow = False
    for child in node.children:
        if not child.is_named and node_text(child, source) == "->":
            saw_arrow = True
        elif saw_arrow and child.type != "block":
            return node_text(child, source)
    return None


def get_signature(node, source: bytes) -> str:
    """Extract function signature (everything before the body block)."""
    parts = []
    for child in node.children:
        if child.type == "block":
            break
        parts.append(node_text(child, source))
    return " ".join(parts)


def is_async_fn(node, source: bytes) -> bool:
    for child in node.children:
        if not child.is_named and node_text(child, source) == "async":
            return True
        if child.type == "identifier" or node_text(child, source) == "fn":
            break
    return False


def get_name(node, source: bytes, name_type: str = "identifier") -> str | None:
    """Get the first child of a specific type as the item name."""
    for child in node.children:
        if child.type == name_type:
            return node_text(child, source)
    return None


def extract_calls(body_node, source: bytes) -> list[str]:
    """Recursively extract function/method names called within a block."""
    calls = []

    def walk(node):
        if node.type == "call_expression":
            func = node.child_by_field_name("function")
            if func is None and node.children:
                func = node.children[0]
            if func:
                if func.type == "identifier":
                    calls.append(node_text(func, source))
                elif func.type == "field_expression":
                    field = func.child_by_field_name("field")
                    if field is None:
                        for c in func.children:
                            if c.type == "field_identifier":
                                field = c
                                break
                    if field:
                        calls.append(node_text(field, source))
                elif func.type == "scoped_identifier":
                    name = func.child_by_field_name("name")
                    if name:
                        calls.append(node_text(name, source))
                    else:
                        parts = node_text(func, source).split("::")
                        if parts:
                            calls.append(parts[-1])
        for child in node.children:
            walk(child)

    walk(body_node)
    return calls


def file_to_module_path(filepath: Path, src_root: Path) -> str:
    rel = filepath.relative_to(src_root)
    parts = list(rel.parts)
    parts[-1] = parts[-1].replace(".rs", "")
    if parts[-1] in ("mod", "lib"):
        parts = parts[:-1]
    if not parts:
        return "crate"
    return "crate::" + "::".join(parts)


# ── Phase 1: Parse ──────────────────────────────────────────────────────


def parse_function(node, source: bytes, module_path: str, file_path: str,
                   is_method: bool = False, owner: str | None = None,
                   impl_is_pymethods: bool = False) -> FunctionInfo:
    name = get_name(node, source, "identifier") or "unknown"
    prefix = f"{module_path}::{owner}" if owner else module_path
    qualified_name = f"{prefix}::{name}"
    attrs = get_attributes(node, source)

    body = None
    for child in node.children:
        if child.type == "block":
            body = child
            break

    return FunctionInfo(
        name=name,
        qualified_name=qualified_name,
        visibility=get_visibility(node),
        is_async=is_async_fn(node, source),
        is_method=is_method,
        signature=get_signature(node, source),
        file_path=file_path,
        line_number=node.start_point[0] + 1,
        docstring=get_doc_comment(node, source),
        return_type=get_return_type(node, source),
        is_pymethod=is_pymethod_fn(attrs, impl_is_pymethods),
        calls=extract_calls(body, source) if body else [],
    )


def parse_file(filepath: Path, src_root: Path, parser: Parser) -> dict:
    """Parse a single .rs file and return extracted entities."""
    source = filepath.read_bytes()
    tree = parser.parse(source)
    root = tree.root_node

    rel_path = str(filepath.relative_to(src_root))
    module_path = file_to_module_path(filepath, src_root)
    loc = source.count(b"\n") + (1 if source and not source.endswith(b"\n") else 0)

    file_info = FileInfo(
        path=rel_path,
        filename=filepath.name,
        loc=loc,
        module_path=module_path,
    )

    functions: list[FunctionInfo] = []
    structs: list[StructInfo] = []
    enums: list[EnumInfo] = []
    traits: list[TraitInfo] = []
    impl_blocks: list[ImplBlock] = []

    for child in root.children:
        if child.type == "function_item":
            functions.append(parse_function(
                child, source, module_path, rel_path,
                is_method=False, owner=None, impl_is_pymethods=False,
            ))

        elif child.type == "struct_item":
            name = get_name(child, source, "type_identifier") or "unknown"
            attrs = get_attributes(child, source)
            structs.append(StructInfo(
                name=name,
                qualified_name=f"{module_path}::{name}",
                visibility=get_visibility(child),
                file_path=rel_path,
                line_number=child.start_point[0] + 1,
                docstring=get_doc_comment(child, source),
                is_pyclass=has_pyclass(attrs),
            ))

        elif child.type == "enum_item":
            name = get_name(child, source, "type_identifier") or "unknown"
            enums.append(EnumInfo(
                name=name,
                qualified_name=f"{module_path}::{name}",
                visibility=get_visibility(child),
                file_path=rel_path,
                line_number=child.start_point[0] + 1,
                docstring=get_doc_comment(child, source),
            ))

        elif child.type == "trait_item":
            name = get_name(child, source, "type_identifier") or "unknown"
            trait_info = TraitInfo(
                name=name,
                qualified_name=f"{module_path}::{name}",
                visibility=get_visibility(child),
                file_path=rel_path,
                line_number=child.start_point[0] + 1,
                docstring=get_doc_comment(child, source),
            )
            traits.append(trait_info)
            # Extract trait method signatures as functions
            for tc in child.children:
                if tc.type == "declaration_list":
                    for item in tc.children:
                        if item.type in ("function_item", "function_signature_item"):
                            functions.append(parse_function(
                                item, source, module_path, rel_path,
                                is_method=True, owner=name,
                                impl_is_pymethods=False,
                            ))

        elif child.type == "impl_item":
            attrs = get_attributes(child, source)
            pymethods = is_pymethods_block(attrs)

            # Determine self_type and optional trait
            type_ids = [c for c in child.children if c.type == "type_identifier"]
            has_for = any(
                not c.is_named and node_text(c, source) == "for"
                for c in child.children
            )

            if has_for and len(type_ids) >= 2:
                trait_name = node_text(type_ids[0], source)
                self_type = node_text(type_ids[1], source)
            elif type_ids:
                trait_name = None
                self_type = node_text(type_ids[0], source)
            else:
                continue

            impl_block = ImplBlock(
                self_type=self_type,
                trait_name=trait_name,
                is_pymethods=pymethods,
            )

            for ic in child.children:
                if ic.type == "declaration_list":
                    for item in ic.children:
                        if item.type == "function_item":
                            fn_info = parse_function(
                                item, source, module_path, rel_path,
                                is_method=True, owner=self_type,
                                impl_is_pymethods=pymethods,
                            )
                            impl_block.methods.append(fn_info)
                            functions.append(fn_info)

            impl_blocks.append(impl_block)

        elif child.type == "use_declaration":
            path_text = None
            for uc in child.children:
                if uc.type == "scoped_identifier":
                    path_text = node_text(uc, source)
                elif uc.type == "use_wildcard":
                    path_text = node_text(uc, source)
                elif uc.type == "scoped_use_list":
                    path_text = node_text(uc, source)
                elif uc.type == "identifier":
                    path_text = node_text(uc, source)
            if path_text and path_text.startswith("crate::"):
                file_info.use_imports.append(path_text)

        elif child.type == "mod_item":
            mod_name = get_name(child, source, "identifier")
            if mod_name:
                file_info.mod_declarations.append(mod_name)

    return {
        "file": file_info,
        "functions": functions,
        "structs": structs,
        "enums": enums,
        "traits": traits,
        "impl_blocks": impl_blocks,
    }


# ── Phase 2: Model ─────────────────────────────────────────────────────


def build_modules(files: list[FileInfo]) -> list[dict]:
    """Build module nodes from file module paths and mod declarations."""
    modules = {}

    for f in files:
        # The file's own module
        if f.module_path not in modules:
            modules[f.module_path] = {
                "qualified_name": f.module_path,
                "name": f.module_path.split("::")[-1] if "::" in f.module_path else "crate",
                "path": f.path,
            }

        # Submodules declared with `mod foo;`
        for mod_name in f.mod_declarations:
            child_path = f"{f.module_path}::{mod_name}"
            if child_path not in modules:
                modules[child_path] = {
                    "qualified_name": child_path,
                    "name": mod_name,
                    "path": "",
                }

    return list(modules.values())


def build_contains_edges(files: list[FileInfo]) -> list[dict]:
    """Build Module CONTAINS Module edges from mod declarations."""
    edges = []
    for f in files:
        for mod_name in f.mod_declarations:
            edges.append({
                "parent": f.module_path,
                "child": f"{f.module_path}::{mod_name}",
            })
    return edges


def build_call_edges(all_functions: list[FunctionInfo],
                     max_targets: int = 5) -> pd.DataFrame:
    """Resolve function calls by name matching.

    Names with more than max_targets definitions are skipped as too ambiguous
    (e.g. new, len, clone exist on dozens of types).
    """
    name_lookup: dict[str, list[str]] = defaultdict(list)
    for fn in all_functions:
        name_lookup[fn.name].append(fn.qualified_name)

    # Build file-path lookup for same-file preference
    qname_to_file: dict[str, str] = {fn.qualified_name: fn.file_path for fn in all_functions}

    edges = []
    seen = set()
    skipped_ambiguous = set()

    for fn in all_functions:
        for called_name in fn.calls:
            if called_name not in name_lookup:
                continue
            targets = name_lookup[called_name]
            if len(targets) > max_targets:
                skipped_ambiguous.add(called_name)
                continue
            for target_qname in targets:
                if target_qname != fn.qualified_name:
                    key = (fn.qualified_name, target_qname)
                    if key not in seen:
                        seen.add(key)
                        edges.append({
                            "caller": fn.qualified_name,
                            "callee": target_qname,
                        })

    if skipped_ambiguous:
        print(f"  Skipped {len(skipped_ambiguous)} ambiguous names "
              f"(>{max_targets} targets): {', '.join(sorted(skipped_ambiguous)[:10])}"
              + (" ..." if len(skipped_ambiguous) > 10 else ""))

    return pd.DataFrame(edges) if edges else pd.DataFrame(columns=["caller", "callee"])


def build_implements_edges(impl_blocks: list[ImplBlock],
                           known_traits: set[str]) -> list[dict]:
    """Build Struct IMPLEMENTS Trait edges (only for traits in our codebase)."""
    edges = []
    seen = set()
    for impl in impl_blocks:
        if impl.trait_name and impl.trait_name in known_traits:
            key = (impl.self_type, impl.trait_name)
            if key not in seen:
                seen.add(key)
                edges.append({
                    "struct_name": impl.self_type,
                    "trait_name": impl.trait_name,
                })
    return edges


def build_has_method_edges(impl_blocks: list[ImplBlock]) -> list[dict]:
    """Build Struct HAS_METHOD Function edges from inherent impl blocks."""
    edges = []
    for impl in impl_blocks:
        if impl.trait_name is None:  # inherent impl only
            for method in impl.methods:
                edges.append({
                    "owner": method.qualified_name.rsplit("::", 1)[0],
                    "method": method.qualified_name,
                })
    return edges


def build_import_edges(files: list[FileInfo], known_modules: set[str]) -> list[dict]:
    """Build File IMPORTS Module edges from use crate::... declarations."""
    edges = []
    for f in files:
        for use_path in f.use_imports:
            # Try progressively shorter prefixes to find a known module
            parts = use_path.split("::")
            for end in range(len(parts), 1, -1):
                candidate = "::".join(parts[:end])
                if candidate in known_modules:
                    edges.append({
                        "file_path": f.path,
                        "module": candidate,
                    })
                    break
    return edges


def build_defines_edges(all_items: list) -> list[dict]:
    """Build File DEFINES item edges, grouped by target type."""
    edges = []
    for item in all_items:
        edges.append({
            "file_path": item.file_path,
            "item_qname": item.qualified_name,
            "item_type": type(item).__name__.replace("Info", ""),
        })
    return edges


# ── Phase 3: Load ──────────────────────────────────────────────────────


def load_graph(files, modules, functions, structs, enums, traits,
               call_edges_df, implements_edges, has_method_edges,
               contains_edges, import_edges, defines_edges) -> kglite.KnowledgeGraph:
    graph = kglite.KnowledgeGraph()

    # -- Nodes --
    if files:
        files_df = pd.DataFrame([{
            "path": f.path, "filename": f.filename, "loc": f.loc,
        } for f in files])
        graph.add_nodes(data=files_df, node_type="File",
                        unique_id_field="path", node_title_field="filename")

    if modules:
        modules_df = pd.DataFrame(modules)
        graph.add_nodes(data=modules_df, node_type="Module",
                        unique_id_field="qualified_name", node_title_field="name")

    if functions:
        funcs_df = pd.DataFrame([{
            "qualified_name": f.qualified_name,
            "name": f.name,
            "visibility": f.visibility,
            "is_async": f.is_async,
            "is_method": f.is_method,
            "signature": f.signature,
            "file_path": f.file_path,
            "line_number": f.line_number,
            "docstring": f.docstring,
            "return_type": f.return_type,
            "is_pymethod": f.is_pymethod,
        } for f in functions])
        graph.add_nodes(data=funcs_df, node_type="Function",
                        unique_id_field="qualified_name", node_title_field="name")

    if structs:
        structs_df = pd.DataFrame([{
            "qualified_name": s.qualified_name,
            "name": s.name,
            "visibility": s.visibility,
            "file_path": s.file_path,
            "line_number": s.line_number,
            "docstring": s.docstring,
            "is_pyclass": s.is_pyclass,
        } for s in structs])
        graph.add_nodes(data=structs_df, node_type="Struct",
                        unique_id_field="qualified_name", node_title_field="name")

    if enums:
        enums_df = pd.DataFrame([{
            "qualified_name": e.qualified_name,
            "name": e.name,
            "visibility": e.visibility,
            "file_path": e.file_path,
            "line_number": e.line_number,
            "docstring": e.docstring,
        } for e in enums])
        graph.add_nodes(data=enums_df, node_type="Enum",
                        unique_id_field="qualified_name", node_title_field="name")

    if traits:
        traits_df = pd.DataFrame([{
            "qualified_name": t.qualified_name,
            "name": t.name,
            "visibility": t.visibility,
            "file_path": t.file_path,
            "line_number": t.line_number,
            "docstring": t.docstring,
        } for t in traits])
        graph.add_nodes(data=traits_df, node_type="Trait",
                        unique_id_field="qualified_name", node_title_field="name")

    # -- Edges --

    # DEFINES: File -> items (separate call per target type)
    if defines_edges:
        for target_type in ("Function", "Struct", "Enum", "Trait"):
            subset = [e for e in defines_edges if e["item_type"] == target_type]
            if subset:
                df = pd.DataFrame(subset)
                graph.add_connections(
                    data=df, connection_type="DEFINES",
                    source_type="File", source_id_field="file_path",
                    target_type=target_type, target_id_field="item_qname",
                )

    # CONTAINS: Module -> Module
    if contains_edges:
        df = pd.DataFrame(contains_edges)
        graph.add_connections(
            data=df, connection_type="CONTAINS",
            source_type="Module", source_id_field="parent",
            target_type="Module", target_id_field="child",
        )

    # CALLS: Function -> Function
    if len(call_edges_df) > 0:
        graph.add_connections(
            data=call_edges_df, connection_type="CALLS",
            source_type="Function", source_id_field="caller",
            target_type="Function", target_id_field="callee",
        )

    # IMPLEMENTS: Struct -> Trait
    if implements_edges:
        df = pd.DataFrame(implements_edges)
        graph.add_connections(
            data=df, connection_type="IMPLEMENTS",
            source_type="Struct", source_id_field="struct_name",
            target_type="Trait", target_id_field="trait_name",
        )

    # HAS_METHOD: Struct -> Function
    if has_method_edges:
        df = pd.DataFrame(has_method_edges)
        graph.add_connections(
            data=df, connection_type="HAS_METHOD",
            source_type="Struct", source_id_field="owner",
            target_type="Function", target_id_field="method",
        )

    # IMPORTS: File -> Module
    if import_edges:
        df = pd.DataFrame(import_edges)
        graph.add_connections(
            data=df, connection_type="IMPORTS",
            source_type="File", source_id_field="file_path",
            target_type="Module", target_id_field="module",
        )

    return graph


# ── Phase 4: Demo queries ──────────────────────────────────────────────


def run_demo_queries(graph: kglite.KnowledgeGraph):
    # Schema overview
    print("\n=== Schema Overview ===")
    schema = graph.schema()
    print(f"Nodes: {schema['node_count']}, Edges: {schema['edge_count']}")
    for name, info in schema["node_types"].items():
        print(f"  :{name} ({info['count']} nodes)")
    for name, info in schema["connection_types"].items():
        print(f"  -[:{name}]- ({info['count']} edges)")

    # PyO3 API surface
    print("\n=== PyO3 API Surface ===")
    result = graph.cypher("""
        MATCH (s:Struct)-[:HAS_METHOD]->(f:Function)
        WHERE s.is_pyclass = true AND f.is_pymethod = true
        RETURN s.name AS struct, count(f) AS methods
        ORDER BY methods DESC
    """)
    for row in result:
        print(f"  {row['struct']}: {row['methods']} Python methods")

    # Most-called functions
    print("\n=== Most-Called Functions (top 10) ===")
    result = graph.cypher("""
        MATCH (caller:Function)-[:CALLS]->(f:Function)
        RETURN f.name AS function, count(caller) AS callers
        ORDER BY callers DESC
        LIMIT 10
    """)
    for row in result:
        print(f"  {row['function']}: called by {row['callers']} functions")

    # Largest files
    print("\n=== Largest Files by Definitions ===")
    result = graph.cypher("""
        MATCH (f:File)-[:DEFINES]->(item)
        RETURN f.filename AS file, f.loc AS lines, count(item) AS definitions
        ORDER BY definitions DESC
        LIMIT 10
    """)
    for row in result:
        print(f"  {row['file']}: {row['definitions']} items, {row['lines']} lines")

    # Multi-hop: functions calling into cypher module
    print("\n=== Functions Calling Into Cypher Module ===")
    result = graph.cypher("""
        MATCH (f:Function)-[:CALLS]->(g:Function)
        WHERE g.file_path CONTAINS 'cypher'
          AND NOT f.file_path CONTAINS 'cypher'
        RETURN DISTINCT f.name AS caller, f.file_path AS file,
               collect(DISTINCT g.name) AS cypher_fns
        ORDER BY caller
        LIMIT 15
    """)
    for row in result:
        fns = ", ".join(row["cypher_fns"][:5])
        if len(row["cypher_fns"]) > 5:
            fns += f" (+{len(row['cypher_fns'])-5} more)"
        print(f"  {row['caller']} ({row['file']}) -> {fns}")

    # Module imports
    print("\n=== Module Import Map ===")
    result = graph.cypher("""
        MATCH (f:File)-[:IMPORTS]->(m:Module)
        RETURN f.filename AS file, collect(DISTINCT m.name) AS imports
        ORDER BY file
        LIMIT 15
    """)
    for row in result:
        print(f"  {row['file']} -> {', '.join(row['imports'])}")

    # PageRank (all edges)
    print("\n=== PageRank: All Edges (top 10) ===")
    pr = graph.pagerank(top_k=10)
    for row in pr:
        print(f"  {row['title']} ({row['type']}): {row['score']:.4f}")

    # PageRank filtered to CALLS edges only
    print("\n=== PageRank: Call Graph Only (top 10) ===")
    pr_calls = graph.pagerank(connection_types=["CALLS"], top_k=10)
    for row in pr_calls:
        print(f"  {row['title']} ({row['type']}): {row['score']:.4f}")

    # PageRank filtered to HAS_METHOD edges only
    print("\n=== PageRank: Method Ownership Only (top 10) ===")
    pr_methods = graph.pagerank(connection_types=["HAS_METHOD"], top_k=10)
    for row in pr_methods:
        print(f"  {row['title']} ({row['type']}): {row['score']:.4f}")

    # Betweenness centrality on CALLS graph
    print("\n=== Betweenness Centrality: Call Graph (top 10) ===")
    bc = graph.betweenness_centrality(connection_types=["CALLS"], top_k=10)
    for row in bc:
        print(f"  {row['title']} ({row['type']}): {row['score']:.4f}")

    # Connection type metadata (multi-pair support)
    print("\n=== Connection Type Details ===")
    for name, info in schema["connection_types"].items():
        src = info.get("source_types", [])
        tgt = info.get("target_types", [])
        pairs = f"{', '.join(sorted(src))} -> {', '.join(sorted(tgt))}"
        multi = " [multi-pair]" if len(src) > 1 or len(tgt) > 1 else ""
        print(f"  :{name} ({info['count']} edges) {pairs}{multi}")

    # Pattern matching with edge bindings
    print("\n=== Pattern: Struct -> Method -> Callee (via match_pattern) ===")
    matches = graph.match_pattern(
        "(s:Struct)-[h:HAS_METHOD]->(m:Function)-[c:CALLS]->(t:Function)"
    )
    for m in matches[:5]:
        print(f"  {m['s']['title']}.{m['m']['title']}() --CALLS--> {m['t']['title']}()")
    if len(matches) > 5:
        print(f"  ... ({len(matches)} total matches)")


# ── Main ────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) > 1:
        src_root = Path(sys.argv[1]).resolve()
    else:
        src_root = Path(__file__).resolve().parent.parent / "src"

    if not src_root.is_dir():
        print(f"Error: {src_root} is not a directory")
        sys.exit(1)

    print(f"Parsing Rust source in: {src_root}")

    parser = Parser(RUST_LANGUAGE)
    rs_files = sorted(src_root.rglob("*.rs"))
    print(f"Found {len(rs_files)} .rs files")

    # Phase 1: Parse all files
    all_files: list[FileInfo] = []
    all_functions: list[FunctionInfo] = []
    all_structs: list[StructInfo] = []
    all_enums: list[EnumInfo] = []
    all_traits: list[TraitInfo] = []
    all_impl_blocks: list[ImplBlock] = []

    for filepath in rs_files:
        result = parse_file(filepath, src_root, parser)
        all_files.append(result["file"])
        all_functions.extend(result["functions"])
        all_structs.extend(result["structs"])
        all_enums.extend(result["enums"])
        all_traits.extend(result["traits"])
        all_impl_blocks.extend(result["impl_blocks"])

    print(f"Parsed: {len(all_functions)} functions, {len(all_structs)} structs, "
          f"{len(all_enums)} enums, {len(all_traits)} traits")

    # Phase 2: Model
    modules = build_modules(all_files)
    known_modules = {m["qualified_name"] for m in modules}
    known_traits = {t.name for t in all_traits}

    contains_edges = build_contains_edges(all_files)
    call_edges_df = build_call_edges(all_functions)
    implements_edges = build_implements_edges(all_impl_blocks, known_traits)
    has_method_edges = build_has_method_edges(all_impl_blocks)
    import_edges = build_import_edges(all_files, known_modules)
    defines_edges = build_defines_edges(
        [i for i in all_functions if not i.is_method]
        + all_structs + all_enums + all_traits,
    )

    print(f"Resolved: {len(call_edges_df)} call edges, {len(has_method_edges)} method edges, "
          f"{len(implements_edges)} impl edges, {len(import_edges)} import edges, "
          f"{len(contains_edges)} module edges, {len(defines_edges)} defines edges")

    # Phase 3: Load
    graph = load_graph(
        all_files, modules, all_functions, all_structs, all_enums, all_traits,
        call_edges_df, implements_edges, has_method_edges,
        contains_edges, import_edges, defines_edges,
    )

    # Phase 4: Demo queries
    run_demo_queries(graph)

    # Save
    output = "kglite_codebase.kgl"
    graph.save(output)
    print(f"\nGraph saved to {output}")
    print(f"Load with: graph = kglite.load('{output}')")


if __name__ == "__main__":
    main()
