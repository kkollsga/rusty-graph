"""
Build a knowledge graph of a multi-language codebase using tree-sitter + KGLite.

Parses source files (Rust, Python, TypeScript/JavaScript), extracts code entities
(functions, classes/structs, enums, traits/interfaces, modules), and loads them
into a KGLite knowledge graph with relationships (DEFINES, CALLS, IMPLEMENTS,
EXTENDS, HAS_METHOD, HAS_SUBMODULE, IMPORTS).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import kglite

from .parsers import (
    ParseResult, FileInfo, FunctionInfo, TypeRelationship,
    get_parser, get_parsers_for_directory,
)
from .parsers.base import extract_parameters_from_signature
from .parsers.models import ProjectInfo, SourceRoot
from .parsers.manifest import read_manifest

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


def _reprefix(value: str, old_prefix: str, new_prefix: str, sep: str) -> str:
    """Replace a module path prefix in a qualified name.

    Returns value unchanged if it doesn't start with old_prefix
    (e.g. short names like base class names).
    """
    if value == old_prefix:
        return new_prefix
    if value.startswith(old_prefix + sep):
        return new_prefix + value[len(old_prefix):]
    return value


def _parse_all_roots(
    project_root: Path,
    source_roots: list[SourceRoot],
    verbose: bool = False,
) -> tuple[ParseResult, frozenset[str]]:
    """Parse source files from specific source roots (manifest-guided)."""
    combined = ParseResult()
    all_noise: set[str] = set()
    parsed_dirs: list[Path] = []  # track parsed roots to detect overlaps

    for root in source_roots:
        if not root.path.is_dir():
            if verbose:
                print(f"  Skipping missing source root: {root.path}")
            continue

        # Skip if this root is a subdirectory of an already-parsed root
        # (e.g. xarray/tests/ is inside xarray/).  Instead, just apply
        # the is_test flag to matching entities already in combined.
        already_covered = any(
            root.path != d and root.path.is_relative_to(d)
            for d in parsed_dirs
        )
        if already_covered:
            if root.is_test:
                rel_prefix = str(root.path.relative_to(project_root))
                for f in combined.files:
                    if f.path.startswith(rel_prefix):
                        f.is_test = True
                for fn in combined.functions:
                    if fn.file_path.startswith(rel_prefix):
                        fn.metadata["is_test"] = True
            if verbose:
                label = root.path.relative_to(project_root)
                print(f"  Skipping {label}/ (already covered by parent root)")
            continue

        if root.language:
            try:
                parsers = [get_parser(root.language)]
            except (ValueError, ImportError) as e:
                if verbose:
                    print(f"  Warning: Skipping {root.language} ({e})")
                continue
        else:
            parsers = get_parsers_for_directory(root.path)

        for parser in parsers:
            label = root.path.relative_to(project_root)
            if verbose:
                print(f"Parsing {parser.language_name} files in {label}/...")
            result = parser.parse_directory(root.path)

            # Adjust all file paths to be relative to project_root
            # (parsers produce paths relative to the source root)
            path_map: dict[str, str] = {}
            for f in result.files:
                old_path = f.path
                abs_path = root.path / old_path
                new_path = str(abs_path.relative_to(project_root))
                path_map[old_path] = new_path
                f.path = new_path
                if root.is_test:
                    f.is_test = True

            # Remap file_path on all entities
            for fn in result.functions:
                fn.file_path = path_map.get(fn.file_path, fn.file_path)
                if root.is_test:
                    fn.metadata["is_test"] = True
            for c in result.classes:
                c.file_path = path_map.get(c.file_path, c.file_path)
            for e in result.enums:
                e.file_path = path_map.get(e.file_path, e.file_path)
            for i in result.interfaces:
                i.file_path = path_map.get(i.file_path, i.file_path)
            for a in result.attributes:
                a.file_path = path_map.get(a.file_path, a.file_path)
            for co in result.constants:
                co.file_path = path_map.get(co.file_path, co.file_path)

            # Remap module_path / qualified_name when source root is nested
            # inside the package (e.g. xarray/tests/ as a separate root).
            # The parser uses src_root.name ("tests") as prefix, but the
            # correct prefix is the full path relative to project_root
            # ("xarray.tests" for Python).
            sep = _get_separator(parser.language_name)
            rel_parts = root.path.relative_to(project_root).parts
            correct_prefix = sep.join(rel_parts)
            parser_prefix = root.path.name
            if correct_prefix != parser_prefix:
                for f in result.files:
                    f.module_path = _reprefix(
                        f.module_path, parser_prefix, correct_prefix, sep)
                for fn in result.functions:
                    fn.qualified_name = _reprefix(
                        fn.qualified_name, parser_prefix, correct_prefix, sep)
                for c in result.classes:
                    c.qualified_name = _reprefix(
                        c.qualified_name, parser_prefix, correct_prefix, sep)
                for e in result.enums:
                    e.qualified_name = _reprefix(
                        e.qualified_name, parser_prefix, correct_prefix, sep)
                for i in result.interfaces:
                    i.qualified_name = _reprefix(
                        i.qualified_name, parser_prefix, correct_prefix, sep)
                for a in result.attributes:
                    a.qualified_name = _reprefix(
                        a.qualified_name, parser_prefix, correct_prefix, sep)
                    a.owner_qualified_name = _reprefix(
                        a.owner_qualified_name, parser_prefix, correct_prefix, sep)
                for co in result.constants:
                    co.qualified_name = _reprefix(
                        co.qualified_name, parser_prefix, correct_prefix, sep)
                for tr in result.type_relationships:
                    tr.source_type = _reprefix(
                        tr.source_type, parser_prefix, correct_prefix, sep)
                    if tr.target_type:
                        tr.target_type = _reprefix(
                            tr.target_type, parser_prefix, correct_prefix, sep)
                    for m in tr.methods:
                        m.qualified_name = _reprefix(
                            m.qualified_name, parser_prefix, correct_prefix, sep)

            combined.merge(result)
            all_noise.update(parser.noise_names)

        parsed_dirs.append(root.path)

    if not combined.files:
        raise FileNotFoundError(
            f"No supported source files found in {project_root}"
        )
    return combined, frozenset(all_noise)


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
    # Map module_path → file_path from parsed files
    file_module_paths: dict[str, str] = {}
    for f in files:
        file_module_paths[f.module_path] = f.path

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
                # Try to resolve path from a parsed file with this module_path
                child_file_path = file_module_paths.get(child_path, "")
                if not child_file_path:
                    # Infer directory path from the parent file's directory
                    parent_dir = "/".join(f.path.split("/")[:-1])
                    if parent_dir:
                        child_file_path = f"{parent_dir}/{sub_name}/"
                modules[child_path] = {
                    "qualified_name": child_path,
                    "name": sub_name,
                    "path": child_file_path,
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


def _infer_lang_group(qualified_name: str) -> str:
    """Infer language group from qualified name separator convention."""
    if "::" in qualified_name:
        return "rust_cpp"
    elif "/" in qualified_name:
        return "go_ts_js"
    return "python_java"


def _build_call_edges(all_functions: list[FunctionInfo],
                      max_targets: int = 5,
                      excluded_names: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Resolve function calls using tiered scope-aware name matching.

    Resolution priority (first match wins):
      1. Receiver hint — "Receiver.method" narrows to targets whose owner matches
      2. Same owner — caller and target share the same qualified_name prefix
      3. Same file — caller and target are defined in the same source file
      4. Same language — caller and target use the same separator convention
      5. Global fallback — all targets with matching bare name

    Names in excluded_names are skipped (common stdlib methods).
    Names with more than max_targets definitions are skipped as too ambiguous.

    Note: Receiver hints are syntactic (field names, not resolved types).
    A call like ``self.inner.method()`` produces hint ``"inner"``, which
    won't match a target owned by the actual type (e.g. ``DirGraph``).
    These calls will fall through to tier 2-5 resolution or remain
    unresolved.  This is an inherent limitation of tree-sitter based
    parsing — resolving field names to types requires type inference.
    """
    name_lookup: dict[str, list[str]] = defaultdict(list)
    for fn in all_functions:
        name_lookup[fn.name].append(fn.qualified_name)

    # Map qualified name -> owner short name for receiver-aware matching.
    # e.g. "crate::server::Server::start" -> "Server"
    qname_to_owner: dict[str, str] = {}
    # Map qualified name -> owner prefix for same-owner matching.
    # e.g. "crate::server::Server::start" -> "crate::server::Server"
    qname_to_prefix: dict[str, str] = {}
    for fn in all_functions:
        qn = fn.qualified_name
        for sep in ("::", ".", "/"):
            if sep in qn:
                owner_path = qn.rsplit(sep, 1)[0]
                qname_to_prefix[qn] = owner_path
                for sep2 in ("::", ".", "/"):
                    if sep2 in owner_path:
                        qname_to_owner[qn] = owner_path.rsplit(sep2, 1)[-1]
                        break
                else:
                    qname_to_owner[qn] = owner_path
                break

    # Map qualified name -> file path for same-file matching.
    qname_to_file: dict[str, str] = {
        fn.qualified_name: fn.file_path for fn in all_functions
    }

    # Accumulate call-site line numbers per (caller, callee) pair
    seen: dict[tuple[str, str], list[int]] = {}

    for fn in all_functions:
        for called_name, call_line in fn.calls:
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

            # Tier 0: Receiver hint — "Receiver.method" narrows by owner
            if receiver_hint:
                filtered = [t for t in targets
                            if qname_to_owner.get(t) == receiver_hint]
                if filtered:
                    targets = filtered

            # Tier 1: Same owner — caller and target share qualified prefix
            if len(targets) > 1:
                caller_prefix = qname_to_prefix.get(fn.qualified_name, "")
                if caller_prefix:
                    same_owner = [t for t in targets
                                  if qname_to_prefix.get(t, "") == caller_prefix]
                    if same_owner:
                        targets = same_owner

            # Tier 2: Same file
            if len(targets) > 1:
                caller_file = fn.file_path
                same_file = [t for t in targets
                             if qname_to_file.get(t) == caller_file]
                if same_file:
                    targets = same_file

            # Tier 3: Same language (infer from separator convention)
            if len(targets) > 1:
                caller_lang = _infer_lang_group(fn.qualified_name)
                same_lang = [t for t in targets
                             if _infer_lang_group(t) == caller_lang]
                if same_lang:
                    targets = same_lang

            if len(targets) > max_targets:
                continue

            for target_qname in targets:
                if target_qname != fn.qualified_name:
                    key = (fn.qualified_name, target_qname)
                    seen.setdefault(key, []).append(call_line)

    edges = []
    for (caller, callee), lines in seen.items():
        sorted_lines = sorted(set(lines))
        edges.append({
            "caller": caller,
            "callee": callee,
            "call_lines": ",".join(str(ln) for ln in sorted_lines),
            "call_count": len(sorted_lines),
        })

    return pd.DataFrame(edges) if edges else pd.DataFrame(
        columns=["caller", "callee", "call_lines", "call_count"]
    )


def _build_type_relationship_edges(
    type_rels: list[TypeRelationship],
    known_interfaces: set[str],
    name_to_qname: dict[str, str],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Build implements, extends, and has_method edges from TypeRelationships.

    Returns (implements_edges, extends_edges, has_method_edges, external_traits).
    External traits are those referenced in impl blocks but not defined locally.
    """
    implements_edges = []
    extends_edges = []
    has_method_edges = []
    seen_impl = set()
    seen_ext = set()
    external_traits: dict[str, dict] = {}  # name -> node dict

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
            key = (tr.source_type, tr.target_type)
            if key not in seen_impl:
                seen_impl.add(key)
                resolved_target = resolve(tr.target_type)
                implements_edges.append({
                    "type_name": resolve(tr.source_type),
                    "interface_name": resolved_target,
                })
                # Track external traits that need node auto-creation
                if tr.target_type not in known_interfaces:
                    if resolved_target not in external_traits:
                        external_traits[resolved_target] = {
                            "qualified_name": resolved_target,
                            "name": tr.target_type,
                            "is_external": True,
                        }
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

    return implements_edges, extends_edges, has_method_edges, list(external_traits.values())


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
                contains_edges, import_edges, defines_edges,
                uses_type_edges, external_traits=None,
                ffi_exposes_edges=None,
                project_info: ProjectInfo | None = None) -> kglite.KnowledgeGraph:
    graph = kglite.KnowledgeGraph()

    # -- Project & Dependency nodes (from manifest) --
    if project_info is not None:
        proj_df = pd.DataFrame([{
            "name": project_info.name,
            "version": project_info.version,
            "description": project_info.description,
            "languages": ", ".join(project_info.languages) or None,
            "authors": ", ".join(project_info.authors) or None,
            "license": project_info.license,
            "repository": project_info.repository_url,
            "build_system": project_info.build_system,
            "manifest": project_info.manifest_path,
        }])
        graph.add_nodes(data=proj_df, node_type="Project",
                        unique_id_field="name", node_title_field="name")

        if project_info.dependencies:
            deps_df = pd.DataFrame([{
                "dep_id": f"{d.name}::{d.group}" if d.group else d.name,
                "name": d.name,
                "version_spec": d.version_spec,
                "is_dev": True if d.is_dev else None,
                "is_optional": True if d.is_optional else None,
                "group": d.group,
            } for d in project_info.dependencies])
            graph.add_nodes(data=deps_df, node_type="Dependency",
                            unique_id_field="dep_id", node_title_field="name")

    # Pre-compute attribute lookup for embedding as JSON on parent nodes
    attrs_by_owner: dict[str, list[dict]] = defaultdict(list)
    for attr in result.attributes:
        entry: dict = {"name": attr.name}
        if attr.type_annotation:
            entry["type"] = attr.type_annotation
        if attr.visibility:
            entry["visibility"] = attr.visibility
        if attr.default_value:
            entry["default"] = attr.default_value
        attrs_by_owner[attr.owner_qualified_name].append(entry)

    # -- Nodes --
    if result.files:
        files_df = pd.DataFrame([{
            "path": f.path, "filename": f.filename, "loc": f.loc,
            "language": f.language,
            "is_test": True if f.is_test else None,
            "annotations": json.dumps(f.annotations) if f.annotations else None,
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
            "is_async": True if f.is_async else None,
            "is_method": True if f.is_method else None,
            "signature": f.signature,
            "file_path": f.file_path,
            "line_number": f.line_number,
            "end_line": f.end_line,
            "docstring": f.docstring,
            "return_type": f.return_type,
            "decorators": ", ".join(f.decorators) if f.decorators else None,
            "parameters": extract_parameters_from_signature(f.signature),
            "type_parameters": f.type_parameters,
            **{k: v for k, v in f.metadata.items()
               if not isinstance(v, (list, dict)) and v is not False},
        } for f in result.functions])
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
                "end_line": c.end_line,
                "docstring": c.docstring,
                "type_parameters": c.type_parameters,
                "fields": json.dumps(attrs_by_owner[c.qualified_name])
                          if c.qualified_name in attrs_by_owner else None,
                **{k: v for k, v in c.metadata.items()
                   if v is not False},
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
            "end_line": e.end_line,
            "docstring": e.docstring,
            "variants": ", ".join(e.variants) if e.variants else None,
            "variant_details": json.dumps(e.variant_details)
                              if e.variant_details else None,
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
                "end_line": i.end_line,
                "docstring": i.docstring,
                "type_parameters": i.type_parameters,
                "fields": json.dumps(attrs_by_owner[i.qualified_name])
                          if i.qualified_name in attrs_by_owner else None,
            } for i in subset])
            graph.add_nodes(data=df, node_type=node_type,
                            unique_id_field="qualified_name",
                            node_title_field="name")

    # Auto-create Trait nodes for external traits referenced in impl blocks
    if external_traits:
        df = pd.DataFrame(external_traits)
        graph.add_nodes(data=df, node_type="Trait",
                        unique_id_field="qualified_name",
                        node_title_field="name",
                        conflict_handling="skip")

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
            columns=["call_lines", "call_count"],
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

    # -- USES_TYPE edges --
    if uses_type_edges:
        for target_type, rows in uses_type_edges.items():
            df = pd.DataFrame(rows)
            graph.add_connections(
                data=df, connection_type="USES_TYPE",
                source_type="Function", source_id_field="function",
                target_type=target_type, target_id_field="type_name",
            )

    # -- FFI EXPOSES edges --
    if ffi_exposes_edges:
        by_target_type: dict[str, list] = defaultdict(list)
        for edge in ffi_exposes_edges:
            by_target_type[edge["target_type"]].append({
                "module_fn": edge["module_fn"],
                "target_qname": edge["target_qname"],
            })
        for target_type, rows in by_target_type.items():
            df = pd.DataFrame(rows)
            graph.add_connections(
                data=df, connection_type="EXPOSES",
                source_type="Function", source_id_field="module_fn",
                target_type=target_type, target_id_field="target_qname",
            )

    # -- Project manifest edges --
    if project_info is not None:
        if project_info.dependencies:
            dep_edges_df = pd.DataFrame([{
                "project": project_info.name,
                "dep_id": f"{d.name}::{d.group}" if d.group else d.name,
            } for d in project_info.dependencies])
            graph.add_connections(
                data=dep_edges_df, connection_type="DEPENDS_ON",
                source_type="Project", source_id_field="project",
                target_type="Dependency", target_id_field="dep_id",
            )
        if result.files:
            has_source_df = pd.DataFrame([{
                "project": project_info.name,
                "file_path": f.path,
            } for f in result.files])
            graph.add_connections(
                data=has_source_df, connection_type="HAS_SOURCE",
                source_type="Project", source_id_field="project",
                target_type="File", target_id_field="file_path",
            )

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


def _build_uses_type_edges(
    functions: list[FunctionInfo],
    classes: list,
    enums: list,
    interfaces: list,
) -> dict[str, list[dict]]:
    """Build USES_TYPE edges from functions to types referenced in signatures.

    Scans each function's parameters and return_type for known type names,
    producing (Function)-[:USES_TYPE]->(Struct|Class|Enum|Trait|...) edges.
    """
    # Collect all known type names → (qualified_name, node_type)
    type_lookup: dict[str, tuple[str, str]] = {}
    for c in classes:
        if len(c.name) > 1:  # skip single-char generics
            type_lookup[c.name] = (c.qualified_name, NODE_TYPE_MAP[c.kind])
    for e in enums:
        if len(e.name) > 1:
            type_lookup[e.name] = (e.qualified_name, "Enum")
    for i in interfaces:
        if len(i.name) > 1:
            type_lookup[i.name] = (i.qualified_name, NODE_TYPE_MAP[i.kind])

    if not type_lookup:
        return {}

    # Build regex matching any known type name as a whole word
    escaped = [re.escape(name) for name in sorted(type_lookup, key=len, reverse=True)]
    pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b")

    # Scan function signatures for type references
    edges_by_target_type: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    for fn in functions:
        text_parts = []
        if fn.signature:
            text_parts.append(fn.signature)
        if fn.return_type:
            text_parts.append(fn.return_type)
        if not text_parts:
            continue

        text = " ".join(text_parts)
        for match in pattern.finditer(text):
            type_name = match.group(1)
            qname, node_type = type_lookup[type_name]
            key = (fn.qualified_name, qname)
            if key not in seen:
                seen.add(key)
                edges_by_target_type[node_type].append({
                    "function": fn.qualified_name,
                    "type_name": qname,
                })

    return dict(edges_by_target_type)


def _build_ffi_exposes_edges(result: ParseResult) -> list[dict]:
    """Build EXPOSES edges from #[pymodule] functions to #[pyclass]/#[pyfunction] items.

    Connects the FFI module entry point to the types and functions it registers,
    showing the cross-language boundary for Maturin/PyO3 projects.
    """
    # Find pymodule functions
    pymodule_fns = [f for f in result.functions
                    if f.metadata.get("is_pymodule")]
    if not pymodule_fns:
        return []

    # Find all pyclass structs and pyfunction functions
    pyclass_qnames = {}
    for c in result.classes:
        if c.metadata.get("is_pyclass"):
            py_name = c.metadata.get("py_name", c.name)
            pyclass_qnames[c.name] = (c.qualified_name, "Struct", py_name)

    pyfunction_qnames = {}
    for f in result.functions:
        if f.metadata.get("ffi_kind") == "pyo3" and not f.is_method and not f.metadata.get("is_pymodule"):
            py_name = f.metadata.get("py_name", f.name)
            pyfunction_qnames[f.name] = (f.qualified_name, "Function", py_name)

    edges = []
    for mod_fn in pymodule_fns:
        # Connect module to all pyclass and pyfunction items
        for qname, target_type, py_name in pyclass_qnames.values():
            edges.append({
                "module_fn": mod_fn.qualified_name,
                "target_qname": qname,
                "target_type": target_type,
                "py_name": py_name,
            })
        for qname, target_type, py_name in pyfunction_qnames.values():
            edges.append({
                "module_fn": mod_fn.qualified_name,
                "target_qname": qname,
                "target_type": target_type,
                "py_name": py_name,
            })

    return edges


# ── Public API ──────────────────────────────────────────────────────────


def build(
    src_dir: str | Path,
    *,
    save_to: str | Path | None = None,
    verbose: bool = False,
    include_tests: bool = True,
) -> kglite.KnowledgeGraph:
    """Parse a codebase and build a KGLite knowledge graph.

    If a project manifest (pyproject.toml, Cargo.toml) is found, uses it
    to discover source roots and extract project metadata.  Otherwise
    falls back to scanning the entire directory.

    Args:
        src_dir: Path to a source directory or manifest file.
        save_to: Optional path to save the graph as a .kgl file.
        verbose: If True, print progress information.
        include_tests: If True, also parse test directories found in the
            manifest.  Has no effect when no manifest is detected.

    Returns:
        A KnowledgeGraph populated with code entities and relationships.

    Raises:
        FileNotFoundError: If src_dir doesn't exist or contains no
            supported files.

    Example::

        from kglite.code_tree import build

        graph = build(".")                          # auto-detect manifest
        graph = build("pyproject.toml")             # explicit manifest
        graph = build("src", include_tests=True)    # directory fallback
    """
    input_path = Path(src_dir).resolve()

    # Resolve project_root and attempt manifest detection
    project_info: ProjectInfo | None = None
    if input_path.is_file():
        project_root = input_path.parent
        from .parsers.manifest import ManifestReader, MANIFEST_READERS
        for reader_cls in MANIFEST_READERS:
            reader: ManifestReader = reader_cls()
            if reader.manifest_filename == input_path.name:
                project_info = reader.read(input_path, project_root)
                break
        if project_info is None:
            raise FileNotFoundError(
                f"Not a recognised manifest file: {input_path.name}"
            )
    elif input_path.is_dir():
        project_root = input_path
        project_info = read_manifest(input_path)
    else:
        raise FileNotFoundError(f"Not a file or directory: {input_path}")

    # Phase 1: Parse
    if project_info and project_info.source_roots:
        roots = list(project_info.source_roots)
        if include_tests:
            roots.extend(project_info.test_roots)
        if verbose:
            root_labels = [
                str(r.path.relative_to(project_root)) for r in roots
            ]
            print(f"Manifest: {project_info.manifest_path} "
                  f"({project_info.build_system})")
            print(f"Source roots: {', '.join(root_labels)}")
        result, noise_names = _parse_all_roots(
            project_root, roots, verbose=verbose,
        )
    else:
        # Fallback: legacy directory scan
        if not project_root.is_dir():
            raise FileNotFoundError(f"Not a directory: {project_root}")
        result, noise_names = _parse_all(project_root, verbose=verbose)

    if verbose:
        print(f"Parsed: {len(result.functions)} functions, "
              f"{len(result.classes)} classes/structs, "
              f"{len(result.enums)} enums, "
              f"{len(result.interfaces)} interfaces/traits, "
              f"{len(result.attributes)} attributes, "
              f"{len(result.constants)} constants")

    # Deduplicate parsed entities — overlapping source/test roots can parse
    # the same file twice.  Last-seen wins so test-root flags take priority.
    def _dedup(items: list, key: str) -> list:
        seen: dict[str, int] = {}
        for idx, item in enumerate(items):
            seen[getattr(item, key)] = idx
        if len(seen) < len(items):
            return [items[i] for i in sorted(seen.values())]
        return items

    result.files = _dedup(result.files, "path")
    result.functions = _dedup(result.functions, "qualified_name")
    result.classes = _dedup(result.classes, "qualified_name")
    result.enums = _dedup(result.enums, "qualified_name")
    result.interfaces = _dedup(result.interfaces, "qualified_name")
    result.constants = _dedup(result.constants, "qualified_name")

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
    implements_edges, extends_edges, has_method_edges, external_traits = \
        _build_type_relationship_edges(
            result.type_relationships, known_interfaces, name_to_qname,
        )
    import_edges = _build_import_edges(result.files, known_modules)
    defines_edges = _build_defines_edges(result)
    uses_type_edges = _build_uses_type_edges(
        result.functions, result.classes, result.enums, result.interfaces,
    )
    ffi_exposes_edges = _build_ffi_exposes_edges(result)

    # Phase 3: Load
    graph = _load_graph(
        result, modules, call_edges_df,
        implements_edges, extends_edges, has_method_edges,
        contains_edges, import_edges, defines_edges,
        uses_type_edges, external_traits, ffi_exposes_edges,
        project_info=project_info,
    )

    if save_to is not None:
        graph.save(str(save_to))
        if verbose:
            print(f"Graph saved to {save_to}")

    return graph
