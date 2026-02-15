"""
Build a knowledge graph of a multi-language codebase using tree-sitter + KGLite.

Parses source files (Rust, Python, TypeScript/JavaScript), extracts code entities
(functions, classes/structs, enums, traits/interfaces, modules), and loads them
into a KGLite knowledge graph with relationships (DEFINES, CALLS, IMPLEMENTS,
EXTENDS, HAS_METHOD, HAS_SUBMODULE, IMPORTS).
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import pandas as pd
import kglite

from .parsers import (
    ParseResult, FileInfo, FunctionInfo, TypeRelationship,
    AttributeInfo, ConstantInfo,
    get_parsers_for_directory,
)
from .parsers.base import extract_parameters_from_signature

# Node type mapping: ClassInfo.kind / InterfaceInfo.kind -> graph node type
NODE_TYPE_MAP = {
    "struct": "Struct",
    "class": "Class",
    "interface": "Interface",
    "trait": "Trait",
    "protocol": "Protocol",
}


# ── Phase 1: Parse (delegated to parsers/) ────────────────────────────


def _parse_all(src_root: Path, verbose: bool = False) -> tuple[ParseResult, frozenset[str]]:
    """Auto-detect languages and parse all source files."""
    parsers = get_parsers_for_directory(src_root)
    if not parsers:
        raise FileNotFoundError(
            f"No supported source files found in {src_root}"
        )

    combined = ParseResult()
    noise_names: set[str] = set()
    for parser in parsers:
        if verbose:
            print(f"Parsing {parser.language_name} files...")
        result = parser.parse_directory(src_root)
        combined.merge(result)
        noise_names.update(parser.noise_names)
    return combined, frozenset(noise_names)


# ── Phase 2: Model ─────────────────────────────────────────────────────


def _get_separator(language: str) -> str:
    if language in ("rust", "cpp"):
        return "::"
    elif language in ("python", "java", "csharp"):
        return "."
    return "/"  # go, c, typescript, javascript


def _build_modules(files: list[FileInfo]) -> list[dict]:
    """Build module nodes from file module paths and submodule declarations."""
    modules = {}
    for f in files:
        sep = _get_separator(f.language)
        path = f.module_path
        if path not in modules:
            parts = path.split(sep)
            modules[path] = {
                "qualified_name": path,
                "name": parts[-1] if parts else path,
                "path": f.path,
                "language": f.language,
            }
        for sub_name in f.submodule_declarations:
            child_path = f"{path}{sep}{sub_name}"
            if child_path not in modules:
                modules[child_path] = {
                    "qualified_name": child_path,
                    "name": sub_name,
                    "path": "",
                    "language": f.language,
                }
    return list(modules.values())


def _build_contains_edges(files: list[FileInfo]) -> list[dict]:
    """Build Module CONTAINS Module edges from submodule declarations."""
    edges = []
    for f in files:
        sep = _get_separator(f.language)
        for sub_name in f.submodule_declarations:
            edges.append({
                "parent": f.module_path,
                "child": f"{f.module_path}{sep}{sub_name}",
            })
    return edges


def _build_call_edges(all_functions: list[FunctionInfo],
                      max_targets: int = 5,
                      excluded_names: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Resolve function calls by name matching with optional receiver hints.

    Calls can be bare names ("process") or qualified ("Receiver.method").
    Qualified calls prefer targets whose owner matches the receiver hint,
    falling back to bare-name matching when no hint match is found.

    Names in excluded_names are skipped (common stdlib methods).
    Names with more than max_targets definitions are skipped as too ambiguous.
    """
    name_lookup: dict[str, list[str]] = defaultdict(list)
    for fn in all_functions:
        name_lookup[fn.name].append(fn.qualified_name)

    # Map qualified name -> owner short name for receiver-aware matching.
    # e.g. "crate::server::Server::start" -> "Server"
    qname_to_owner: dict[str, str] = {}
    for fn in all_functions:
        qn = fn.qualified_name
        for sep in ("::", ".", "/"):
            if sep in qn:
                owner_path = qn.rsplit(sep, 1)[0]
                for sep2 in ("::", ".", "/"):
                    if sep2 in owner_path:
                        qname_to_owner[qn] = owner_path.rsplit(sep2, 1)[-1]
                        break
                else:
                    qname_to_owner[qn] = owner_path
                break

    edges = []
    seen = set()

    for fn in all_functions:
        for called_name in fn.calls:
            # Parse qualified call: "Receiver.method" or bare "method"
            if "." in called_name:
                receiver_hint, method_name = called_name.rsplit(".", 1)
            else:
                receiver_hint = None
                method_name = called_name

            if method_name in excluded_names:
                continue
            if method_name not in name_lookup:
                continue

            targets = name_lookup[method_name]

            # If receiver hint available, prefer targets whose owner matches
            if receiver_hint:
                filtered = [t for t in targets
                            if qname_to_owner.get(t) == receiver_hint]
                if filtered:
                    targets = filtered

            if len(targets) > max_targets:
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

    return pd.DataFrame(edges) if edges else pd.DataFrame(columns=["caller", "callee"])


def _build_type_relationship_edges(
    type_rels: list[TypeRelationship],
    known_interfaces: set[str],
    name_to_qname: dict[str, str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build implements, extends, and has_method edges from TypeRelationships."""
    implements_edges = []
    extends_edges = []
    has_method_edges = []
    seen_impl = set()
    seen_ext = set()

    def resolve(name: str) -> str:
        return name_to_qname.get(name, name)

    for tr in type_rels:
        if tr.relationship == "inherent":
            for method in tr.methods:
                for sep in ("::", "."):
                    if sep in method.qualified_name:
                        owner = method.qualified_name.rsplit(sep, 1)[0]
                        break
                else:
                    owner = method.qualified_name
                has_method_edges.append({
                    "owner": owner,
                    "method": method.qualified_name,
                })
        elif tr.relationship == "implements" and tr.target_type:
            if tr.target_type in known_interfaces:
                key = (tr.source_type, tr.target_type)
                if key not in seen_impl:
                    seen_impl.add(key)
                    implements_edges.append({
                        "type_name": resolve(tr.source_type),
                        "interface_name": resolve(tr.target_type),
                    })
            for method in tr.methods:
                for sep in ("::", "."):
                    if sep in method.qualified_name:
                        owner = method.qualified_name.rsplit(sep, 1)[0]
                        break
                else:
                    owner = method.qualified_name
                has_method_edges.append({
                    "owner": owner,
                    "method": method.qualified_name,
                })
        elif tr.relationship == "extends" and tr.target_type:
            key = (tr.source_type, tr.target_type)
            if key not in seen_ext:
                seen_ext.add(key)
                extends_edges.append({
                    "child_name": resolve(tr.source_type),
                    "parent_name": resolve(tr.target_type),
                })

    return implements_edges, extends_edges, has_method_edges


def _build_import_edges(files: list[FileInfo], known_modules: set[str]) -> list[dict]:
    """Build File IMPORTS Module edges from import declarations."""
    edges = []
    for f in files:
        sep = _get_separator(f.language)
        for use_path in f.imports:
            parts = use_path.split(sep)
            for end in range(len(parts), 0, -1):
                candidate = sep.join(parts[:end])
                if candidate in known_modules:
                    edges.append({
                        "file_path": f.path,
                        "module": candidate,
                    })
                    break
    return edges


def _build_defines_edges(result: ParseResult) -> list[dict]:
    """Build File DEFINES item edges."""
    edges = []
    for fn in result.functions:
        if not fn.is_method:
            edges.append({
                "file_path": fn.file_path,
                "item_qname": fn.qualified_name,
                "item_type": "Function",
            })
    for cls in result.classes:
        edges.append({
            "file_path": cls.file_path,
            "item_qname": cls.qualified_name,
            "item_type": NODE_TYPE_MAP[cls.kind],
        })
    for enum in result.enums:
        edges.append({
            "file_path": enum.file_path,
            "item_qname": enum.qualified_name,
            "item_type": "Enum",
        })
    for iface in result.interfaces:
        edges.append({
            "file_path": iface.file_path,
            "item_qname": iface.qualified_name,
            "item_type": NODE_TYPE_MAP[iface.kind],
        })
    for const in result.constants:
        edges.append({
            "file_path": const.file_path,
            "item_qname": const.qualified_name,
            "item_type": "Constant",
        })
    return edges


# ── Phase 3: Load ──────────────────────────────────────────────────────


def _load_graph(result: ParseResult, modules, call_edges_df,
                implements_edges, extends_edges, has_method_edges,
                contains_edges, import_edges, defines_edges) -> kglite.KnowledgeGraph:
    graph = kglite.KnowledgeGraph()

    # -- Nodes --
    if result.files:
        files_df = pd.DataFrame([{
            "path": f.path, "filename": f.filename, "loc": f.loc,
            "language": f.language,
        } for f in result.files])
        graph.add_nodes(data=files_df, node_type="File",
                        unique_id_field="path", node_title_field="filename")

    if modules:
        modules_df = pd.DataFrame(modules)
        graph.add_nodes(data=modules_df, node_type="Module",
                        unique_id_field="qualified_name", node_title_field="name")

    if result.functions:
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
            "decorators": ", ".join(f.decorators) if f.decorators else None,
            "parameters": extract_parameters_from_signature(f.signature),
            "type_parameters": f.type_parameters,
            **{k: v for k, v in f.metadata.items()
               if not isinstance(v, (list, dict))},
        } for f in result.functions])
        # Fix mixed True/NaN columns from conditional metadata to proper bool
        for col in funcs_df.columns:
            if funcs_df[col].dtype == "object":
                vals = funcs_df[col].dropna().unique()
                if len(vals) > 0 and all(isinstance(v, bool) for v in vals):
                    funcs_df[col] = funcs_df[col].fillna(False).astype(bool)
        graph.add_nodes(data=funcs_df, node_type="Function",
                        unique_id_field="qualified_name", node_title_field="name")

    for kind, node_type in [("struct", "Struct"), ("class", "Class")]:
        subset = [c for c in result.classes if c.kind == kind]
        if subset:
            df = pd.DataFrame([{
                "qualified_name": c.qualified_name,
                "name": c.name,
                "visibility": c.visibility,
                "file_path": c.file_path,
                "line_number": c.line_number,
                "docstring": c.docstring,
                "type_parameters": c.type_parameters,
                **{k: v for k, v in c.metadata.items()},
            } for c in subset])
            graph.add_nodes(data=df, node_type=node_type,
                            unique_id_field="qualified_name",
                            node_title_field="name")

    if result.enums:
        df = pd.DataFrame([{
            "qualified_name": e.qualified_name,
            "name": e.name,
            "visibility": e.visibility,
            "file_path": e.file_path,
            "line_number": e.line_number,
            "docstring": e.docstring,
            "variants": ", ".join(e.variants) if e.variants else None,
        } for e in result.enums])
        graph.add_nodes(data=df, node_type="Enum",
                        unique_id_field="qualified_name", node_title_field="name")

    for kind, node_type in [("trait", "Trait"), ("protocol", "Protocol"),
                             ("interface", "Interface")]:
        subset = [i for i in result.interfaces if i.kind == kind]
        if subset:
            df = pd.DataFrame([{
                "qualified_name": i.qualified_name,
                "name": i.name,
                "visibility": i.visibility,
                "file_path": i.file_path,
                "line_number": i.line_number,
                "docstring": i.docstring,
                "type_parameters": i.type_parameters,
            } for i in subset])
            graph.add_nodes(data=df, node_type=node_type,
                            unique_id_field="qualified_name",
                            node_title_field="name")

    if result.attributes:
        df = pd.DataFrame([{
            "qualified_name": a.qualified_name,
            "name": a.name,
            "owner": a.owner_qualified_name,
            "type_annotation": a.type_annotation,
            "visibility": a.visibility,
            "file_path": a.file_path,
            "line_number": a.line_number,
            "default_value": a.default_value,
        } for a in result.attributes])
        graph.add_nodes(data=df, node_type="Attribute",
                        unique_id_field="qualified_name", node_title_field="name")

    if result.constants:
        df = pd.DataFrame([{
            "qualified_name": c.qualified_name,
            "name": c.name,
            "kind": c.kind,
            "type_annotation": c.type_annotation,
            "value_preview": c.value_preview,
            "visibility": c.visibility,
            "file_path": c.file_path,
            "line_number": c.line_number,
        } for c in result.constants])
        graph.add_nodes(data=df, node_type="Constant",
                        unique_id_field="qualified_name", node_title_field="name")

    # -- Edges --

    if defines_edges:
        target_types = {e["item_type"] for e in defines_edges}
        for target_type in sorted(target_types):
            subset = [e for e in defines_edges if e["item_type"] == target_type]
            if subset:
                df = pd.DataFrame(subset)
                graph.add_connections(
                    data=df, connection_type="DEFINES",
                    source_type="File", source_id_field="file_path",
                    target_type=target_type, target_id_field="item_qname",
                )

    if contains_edges:
        df = pd.DataFrame(contains_edges)
        graph.add_connections(
            data=df, connection_type="HAS_SUBMODULE",
            source_type="Module", source_id_field="parent",
            target_type="Module", target_id_field="child",
        )

    if len(call_edges_df) > 0:
        graph.add_connections(
            data=call_edges_df, connection_type="CALLS",
            source_type="Function", source_id_field="caller",
            target_type="Function", target_id_field="callee",
        )

    if implements_edges:
        df = pd.DataFrame(implements_edges)
        _add_typed_connections(graph, df, "IMPLEMENTS",
                               "type_name", "interface_name",
                               result.classes, result.interfaces)

    if extends_edges:
        df = pd.DataFrame(extends_edges)
        _add_typed_connections_same(graph, df, "EXTENDS",
                                    "child_name", "parent_name",
                                    result.classes)

    if has_method_edges:
        df = pd.DataFrame(has_method_edges)
        _add_has_method_connections(graph, df, result.classes, result.interfaces)

    if import_edges:
        df = pd.DataFrame(import_edges)
        graph.add_connections(
            data=df, connection_type="IMPORTS",
            source_type="File", source_id_field="file_path",
            target_type="Module", target_id_field="module",
        )

    if result.attributes:
        _add_has_attribute_connections(graph, result.attributes,
                                       result.classes, result.interfaces)

    return graph


def _add_typed_connections(graph, df, conn_type, source_field, target_field,
                           classes, interfaces):
    """Add connections where source/target types depend on the entity kind."""
    src_type_map = {}
    for c in classes:
        src_type_map[c.name] = NODE_TYPE_MAP[c.kind]
        src_type_map[c.qualified_name] = NODE_TYPE_MAP[c.kind]

    tgt_type_map = {}
    for i in interfaces:
        tgt_type_map[i.name] = NODE_TYPE_MAP[i.kind]
        tgt_type_map[i.qualified_name] = NODE_TYPE_MAP[i.kind]

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for _, row in df.iterrows():
        src_nt = src_type_map.get(row[source_field], "Struct")
        tgt_nt = tgt_type_map.get(row[target_field], "Trait")
        groups[(src_nt, tgt_nt)].append(row.to_dict())

    for (src_nt, tgt_nt), rows in groups.items():
        sub_df = pd.DataFrame(rows)
        graph.add_connections(
            data=sub_df, connection_type=conn_type,
            source_type=src_nt, source_id_field=source_field,
            target_type=tgt_nt, target_id_field=target_field,
        )


def _add_typed_connections_same(graph, df, conn_type, source_field, target_field,
                                 classes):
    """Add connections between same-type entities (e.g. Class EXTENDS Class)."""
    type_map = {}
    for c in classes:
        type_map[c.name] = NODE_TYPE_MAP[c.kind]
        type_map[c.qualified_name] = NODE_TYPE_MAP[c.kind]

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for _, row in df.iterrows():
        src_nt = type_map.get(row[source_field], "Class")
        tgt_nt = type_map.get(row[target_field], "Class")
        groups[(src_nt, tgt_nt)].append(row.to_dict())

    for (src_nt, tgt_nt), rows in groups.items():
        sub_df = pd.DataFrame(rows)
        graph.add_connections(
            data=sub_df, connection_type=conn_type,
            source_type=src_nt, source_id_field=source_field,
            target_type=tgt_nt, target_id_field=target_field,
        )


def _add_has_method_connections(graph, df, classes, interfaces=None):
    """Add HAS_METHOD connections, routing through correct source node type."""
    type_map = {}
    for c in classes:
        type_map[c.name] = NODE_TYPE_MAP[c.kind]
        type_map[c.qualified_name] = NODE_TYPE_MAP[c.kind]
    if interfaces:
        for i in interfaces:
            type_map[i.name] = NODE_TYPE_MAP[i.kind]
            type_map[i.qualified_name] = NODE_TYPE_MAP[i.kind]

    schema = graph.schema()
    # Pick a default type from what's available
    default_type = None
    for candidate in ("Class", "Struct", "Trait", "Interface", "Protocol"):
        if candidate in schema["node_types"]:
            default_type = candidate
            break
    if default_type is None:
        return

    groups: dict[str, list] = defaultdict(list)
    for _, row in df.iterrows():
        owner = row["owner"]
        for sep in ("::", "."):
            if sep in owner:
                name = owner.rsplit(sep, 1)[-1]
                break
        else:
            name = owner
        src_nt = type_map.get(name, type_map.get(owner, default_type))
        groups[src_nt].append(row.to_dict())

    for src_nt, rows in groups.items():
        if src_nt not in schema["node_types"]:
            continue
        sub_df = pd.DataFrame(rows)
        graph.add_connections(
            data=sub_df, connection_type="HAS_METHOD",
            source_type=src_nt, source_id_field="owner",
            target_type="Function", target_id_field="method",
        )


def _add_has_attribute_connections(graph, attributes, classes, interfaces):
    """Add HAS_ATTRIBUTE connections from owner type to attribute."""
    type_map = {}
    for c in classes:
        type_map[c.name] = NODE_TYPE_MAP[c.kind]
        type_map[c.qualified_name] = NODE_TYPE_MAP[c.kind]
    for i in interfaces:
        type_map[i.name] = NODE_TYPE_MAP[i.kind]
        type_map[i.qualified_name] = NODE_TYPE_MAP[i.kind]

    schema = graph.schema()

    groups: dict[str, list] = defaultdict(list)
    for attr in attributes:
        owner = attr.owner_qualified_name
        for sep in ("::", "."):
            if sep in owner:
                name = owner.rsplit(sep, 1)[-1]
                break
        else:
            name = owner
        src_nt = type_map.get(name, type_map.get(owner, "Class"))
        groups[src_nt].append({
            "owner": owner,
            "attribute": attr.qualified_name,
        })

    for src_nt, rows in groups.items():
        if src_nt not in schema["node_types"]:
            continue
        sub_df = pd.DataFrame(rows)
        graph.add_connections(
            data=sub_df, connection_type="HAS_ATTRIBUTE",
            source_type=src_nt, source_id_field="owner",
            target_type="Attribute", target_id_field="attribute",
        )


# ── Public API ──────────────────────────────────────────────────────────


def build(
    src_dir: str | Path,
    *,
    save_to: str | Path | None = None,
    verbose: bool = False,
) -> kglite.KnowledgeGraph:
    """Parse a codebase and build a KGLite knowledge graph.

    Args:
        src_dir: Path to the source directory to parse.
        save_to: Optional path to save the graph as a .kgl file.
        verbose: If True, print progress information.

    Returns:
        A KnowledgeGraph populated with code entities and relationships.

    Raises:
        FileNotFoundError: If src_dir doesn't exist or contains no supported files.

    Example::

        from kglite.code_tree import build

        graph = build("/path/to/project/src")
        result = graph.cypher("MATCH (f:Function) RETURN f.name LIMIT 10")
    """
    src_root = Path(src_dir).resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {src_root}")

    # Phase 1: Parse
    result, noise_names = _parse_all(src_root, verbose=verbose)

    if verbose:
        print(f"Parsed: {len(result.functions)} functions, "
              f"{len(result.classes)} classes/structs, "
              f"{len(result.enums)} enums, "
              f"{len(result.interfaces)} interfaces/traits, "
              f"{len(result.attributes)} attributes, "
              f"{len(result.constants)} constants")

    # Phase 2: Model
    modules = _build_modules(result.files)
    known_modules = {m["qualified_name"] for m in modules}
    known_interfaces = {i.name for i in result.interfaces}

    name_to_qname: dict[str, str] = {}
    for c in result.classes:
        name_to_qname[c.name] = c.qualified_name
    for i in result.interfaces:
        name_to_qname[i.name] = i.qualified_name

    contains_edges = _build_contains_edges(result.files)
    call_edges_df = _build_call_edges(result.functions, excluded_names=noise_names)
    implements_edges, extends_edges, has_method_edges = \
        _build_type_relationship_edges(
            result.type_relationships, known_interfaces, name_to_qname,
        )
    import_edges = _build_import_edges(result.files, known_modules)
    defines_edges = _build_defines_edges(result)

    # Phase 3: Load
    graph = _load_graph(
        result, modules, call_edges_df,
        implements_edges, extends_edges, has_method_edges,
        contains_edges, import_edges, defines_edges,
    )

    if save_to is not None:
        graph.save(str(save_to))
        if verbose:
            print(f"Graph saved to {save_to}")

    return graph
