"""Project manifest readers for pyproject.toml, Cargo.toml, etc."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models import DependencyInfo, ProjectInfo, SourceRoot

# ── TOML loading (stdlib 3.11+, tomli fallback for 3.10) ─────────────

_tomllib = None


def _load_toml(path: Path) -> dict:
    """Load a TOML file, using stdlib tomllib or tomli fallback."""
    global _tomllib
    if _tomllib is None:
        try:
            import tomllib as _tl
        except ModuleNotFoundError:
            try:
                import tomli as _tl  # type: ignore[no-redef]
            except ImportError:
                raise ImportError(
                    "TOML parsing requires Python 3.11+ or the 'tomli' package. "
                    "Install with: pip install tomli"
                ) from None
        _tomllib = _tl
    with open(path, "rb") as f:
        return _tomllib.load(f)


# ── Base class ────────────────────────────────────────────────────────


class ManifestReader(ABC):
    """Base class for project manifest readers."""

    @property
    @abstractmethod
    def manifest_filename(self) -> str:
        """The filename this reader handles (e.g. 'pyproject.toml')."""
        ...

    @abstractmethod
    def read(self, manifest_path: Path, project_root: Path) -> ProjectInfo:
        """Parse the manifest and return project metadata."""
        ...


# ── pyproject.toml ────────────────────────────────────────────────────


class PyProjectReader(ManifestReader):
    """Reads pyproject.toml (PEP 621) with Maturin/setuptools awareness."""

    @property
    def manifest_filename(self) -> str:
        return "pyproject.toml"

    def read(self, manifest_path: Path, project_root: Path) -> ProjectInfo:
        data = _load_toml(manifest_path)
        project = data.get("project", {})
        build_sys = data.get("build-system", {})
        tool = data.get("tool", {})

        name = project.get("name", project_root.name)
        build_backend = build_sys.get("build-backend", "")

        version = project.get("version")
        if version is None and "version" in project.get("dynamic", []):
            version = "dynamic"

        info = ProjectInfo(
            name=name,
            version=version,
            description=project.get("description"),
            languages=["python"],
            authors=_extract_authors(project.get("authors", [])),
            license=_extract_license(project.get("license")),
            repository_url=_extract_repo_url(project.get("urls", {})),
            manifest_path=str(manifest_path.relative_to(project_root)),
            build_system=build_backend or "unknown",
        )

        # Dependencies
        for dep_str in project.get("dependencies", []):
            info.dependencies.append(_parse_dep_string(dep_str))
        for group, deps in project.get("optional-dependencies", {}).items():
            for dep_str in deps:
                dep = _parse_dep_string(dep_str)
                dep.is_optional = True
                dep.group = group
                info.dependencies.append(dep)

        # Source root: find Python package directory
        py_root = _find_python_package(project_root, name)
        if py_root:
            info.source_roots.append(SourceRoot(
                path=py_root, language="python", label="python-package",
            ))

        # Maturin: chain to Cargo.toml for Rust source roots
        if "maturin" in build_backend:
            cargo_path = project_root / "Cargo.toml"
            if cargo_path.is_file():
                cargo_reader = CargoTomlReader()
                rust_info = cargo_reader.read(cargo_path, project_root)
                info.source_roots.extend(rust_info.source_roots)
                # Skip Cargo test_roots — pytest config handles tests better
                # for Maturin projects (tests/ may contain Python, not Rust)
                info.languages.append("rust")
                # Merge Rust dependencies
                for dep in rust_info.dependencies:
                    info.dependencies.append(dep)
                # Use Cargo version if pyproject has dynamic version
                if info.version is None and rust_info.version:
                    info.version = rust_info.version

        # Test roots from pytest config (deduplicate against Cargo test roots)
        existing_test_paths = {r.path for r in info.test_roots}
        pytest_opts = tool.get("pytest", {}).get("ini_options", {})
        test_paths = pytest_opts.get("testpaths", [])
        for tp in test_paths:
            test_dir = project_root / tp
            if test_dir.is_dir() and test_dir not in existing_test_paths:
                info.test_roots.append(SourceRoot(
                    path=test_dir, is_test=True, label="pytest",
                ))
        # Fallback: check tests/ or test/
        if not info.test_roots:
            for candidate in ("tests", "test"):
                test_dir = project_root / candidate
                if test_dir.is_dir() and test_dir not in existing_test_paths:
                    info.test_roots.append(SourceRoot(
                        path=test_dir, is_test=True, label="test-dir",
                    ))
                    break

        return info


# ── Cargo.toml ────────────────────────────────────────────────────────


class CargoTomlReader(ManifestReader):
    """Reads Cargo.toml for Rust projects and workspaces."""

    @property
    def manifest_filename(self) -> str:
        return "Cargo.toml"

    def read(self, manifest_path: Path, project_root: Path) -> ProjectInfo:
        data = _load_toml(manifest_path)
        package = data.get("package", {})

        info = ProjectInfo(
            name=package.get("name", project_root.name),
            version=package.get("version"),
            description=package.get("description"),
            languages=["rust"],
            authors=package.get("authors", []),
            license=package.get("license"),
            repository_url=package.get("repository"),
            manifest_path=str(manifest_path.relative_to(project_root)),
            build_system="cargo",
        )

        # Source root: src/ directory
        src_dir = project_root / "src"
        if src_dir.is_dir():
            info.source_roots.append(SourceRoot(
                path=src_dir, language="rust", label="rust-crate",
            ))

        # Workspace members
        workspace = data.get("workspace", {})
        for member_glob in workspace.get("members", []):
            for member_dir in sorted(project_root.glob(member_glob)):
                if member_dir.is_dir() and (member_dir / "src").is_dir():
                    info.source_roots.append(SourceRoot(
                        path=member_dir / "src", language="rust",
                        label=f"workspace:{member_dir.name}",
                    ))

        # Test root: tests/ directory (integration tests)
        tests_dir = project_root / "tests"
        if tests_dir.is_dir():
            info.test_roots.append(SourceRoot(
                path=tests_dir, language="rust", is_test=True,
                label="rust-integration-tests",
            ))

        # Dependencies
        for name, spec in data.get("dependencies", {}).items():
            info.dependencies.append(_parse_cargo_dep(name, spec, is_dev=False))
        for name, spec in data.get("dev-dependencies", {}).items():
            info.dependencies.append(_parse_cargo_dep(name, spec, is_dev=True))

        return info


# ── Detection & reading ───────────────────────────────────────────────

MANIFEST_READERS: list[type[ManifestReader]] = [
    PyProjectReader,
    CargoTomlReader,
]


def detect_manifest(project_root: Path) -> ManifestReader | None:
    """Auto-detect the primary manifest in a directory (first match wins)."""
    for reader_cls in MANIFEST_READERS:
        reader = reader_cls()
        if (project_root / reader.manifest_filename).is_file():
            return reader
    return None


def read_manifest(project_root: Path) -> ProjectInfo | None:
    """Auto-detect and read the project manifest, if any."""
    reader = detect_manifest(project_root)
    if reader is None:
        return None
    manifest_path = project_root / reader.manifest_filename
    return reader.read(manifest_path, project_root)


# ── Helpers ───────────────────────────────────────────────────────────


def _find_python_package(project_root: Path, project_name: str) -> Path | None:
    """Find the Python package directory for a project."""
    pkg_name = project_name.replace("-", "_")

    # Flat layout: project_root/package/
    candidate = project_root / pkg_name
    if candidate.is_dir() and (candidate / "__init__.py").is_file():
        return candidate

    # src layout: project_root/src/package/
    candidate = project_root / "src" / pkg_name
    if candidate.is_dir() and (candidate / "__init__.py").is_file():
        return candidate

    return None


def _extract_authors(authors: list) -> list[str]:
    """Extract author names from pyproject.toml authors list."""
    result = []
    for a in authors:
        if isinstance(a, dict):
            name = a.get("name", "")
            email = a.get("email", "")
            result.append(f"{name} <{email}>" if email else name)
        elif isinstance(a, str):
            result.append(a)
    return result


def _extract_license(license_val) -> str | None:
    """Extract license string from pyproject.toml license field."""
    if isinstance(license_val, str):
        return license_val
    if isinstance(license_val, dict):
        return license_val.get("text") or license_val.get("file")
    return None


def _extract_repo_url(urls: dict) -> str | None:
    """Extract repository URL from pyproject.toml urls."""
    for key in ("Repository", "repository", "Source", "source", "Homepage"):
        if key in urls:
            return urls[key]
    return None


def _parse_dep_string(dep_str: str) -> DependencyInfo:
    """Parse a PEP 508 dependency string like 'pandas>=1.5'."""
    # Split on first version specifier character
    for i, ch in enumerate(dep_str):
        if ch in ">=<!~":
            return DependencyInfo(
                name=dep_str[:i].strip(),
                version_spec=dep_str[i:].strip(),
            )
    return DependencyInfo(name=dep_str.strip())


def _parse_cargo_dep(name: str, spec, *, is_dev: bool) -> DependencyInfo:
    """Parse a Cargo.toml dependency entry."""
    if isinstance(spec, str):
        version = spec
    elif isinstance(spec, dict):
        version = spec.get("version")
    else:
        version = None
    return DependencyInfo(
        name=name,
        version_spec=version,
        is_dev=is_dev,
    )
