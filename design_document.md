# Open Source Library Design Document

A reference blueprint for building, documenting, and shipping an open source library. Principles are language-agnostic; concrete examples are included where they make the advice actionable.

---

## 1. Repository Setup & Discoverability

### Repository metadata

- **Description:** One sentence, keyword-rich. Appears in search results, embeds, and social previews.
- **Topics / Tags:** 6–10 tags targeting what users actually search for. Avoid generic tags (`python`, `javascript`, `rust`) — they're too saturated to drive discovery. Prefer domain-specific terms that match how users search for solutions.
- **Social preview image:** Custom 1280×640px image. Shows up when your repo link is shared on Twitter, LinkedIn, Discord, Slack. First visual impression for many visitors.

### README as a landing page

The README is your most important marketing asset. Search engines and platform search heavily weight the title and first paragraph.

**Structure:**

```text
# LibName — Short Tagline with Keywords          ← H1 with search terms
[badges]                                          ← Package registry, docs, license, CI
Opening paragraph with ALL high-value keywords    ← Most important text for SEO
## Why LibName?                                   ← Benefit-oriented bullet points
## Quick Start                                    ← install command + minimal working example
## Use Cases                                      ← 3-4 scenarios with code snippets
## Key Features                                   ← Compact table, not a wall of text
## Documentation                                  ← Top 5 links + pointer to full docs
## Call for Contributions                         ← Welcome contributors (not just code)
## Requirements / License                         ← Reference info at the bottom
```

**Key principles:**

- Title must include searchable keywords, not just the library name
- Opening paragraph must hit every high-value search term naturally (domain, language, technology, use cases)
- "Why X?" section answers "why should I use this?" before showing any code
- Use Cases section maps real search queries to your library's capabilities
- Docs table: 5 links max — README is a landing page, not a sitemap
- Requirements/license go at the bottom — they're reference info, not selling points
- Explicitly welcome non-code contributions (docs, tutorials, bug reports, reviews) — NumPy does this prominently and it drives community engagement

### Lessons learned

- A README rewrite focused on keywords and use cases can improve discoverability more than any code change
- Social preview image is often overlooked but significantly improves click-through on shared links
- A comparison table vs. alternatives helps users decide faster (and ranks for comparison searches)
- Multiple trust/credibility badges (funding, security, adoption metrics) signal project maturity

---

## 2. CI / CD

### Recommended workflow structure

**CI pipeline** — runs on every push/PR to main:

```text
┌─────────────┐    ┌───────────────┐    ┌──────────────────┐
│ Lint/Format  │    │ Language lint  │    │ Tests (matrix)   │
│ (fast, no    │    │ (if multiple  │    │ - version A      │
│  build step) │    │  languages)   │    │ - version B      │
└─────────────┘    └───────────────┘    │ - coverage upload│
                                        └──────────────────┘
```

**Publish pipeline** — runs on push to main, conditional:

```text
┌───────────────┐    ┌──────────┐    ┌───────────────────────┐    ┌─────────┐
│ Version check │───▶│ CI gate  │───▶│ Build artifacts       │───▶│ Publish │
│ (skip if      │    │ (wait for│    │ (platform matrix if   │    │ - Pkg   │
│  already      │    │  ALL CI  │    │  compiled extension)  │    │   registry
│  published)   │    │  checks) │    │                       │    │ - GitHub│
└───────────────┘    └──────────┘    └───────────────────────┘    └─────────┘
```

### Key patterns

- **Version check before building:** Query your package registry's API to see if the version already exists. Skip the entire build if nothing to publish. Saves CI minutes.
- **CI gate before publish:** Wait for ALL checks (lint, tests, coverage) to pass before building release artifacts. Never publish untested or unlinted code.
- **Matrix testing:** Test against multiple language/runtime versions. Use `fail-fast: false` so a failure in one version doesn't cancel the others.
- **Trusted publishing / keyless auth:** Most registries support OIDC-based publishing (PyPI Trusted Publishers, npm provenance, crates.io). No secrets to manage or rotate.
- **Changelog extraction:** Parse your changelog to auto-populate release notes. Avoids manual copy-paste and ensures releases always have notes.
- **Benchmarks:** Two modes. Manual trigger for local comparison (`make bench-save`, `make bench-compare`). Automatic on pushes to main for historical tracking — store results as JSON and use a benchmark action to detect regressions over time. Alert but don't block (`fail-on-alert: false`).
- **Coverage in CI:** Collect coverage during test runs. Upload to a service (Codecov, Coveralls) on one matrix version only to avoid duplicates.

#### Example: PyPI version check (GitHub Actions)

```yaml
- name: Check version against PyPI
  run: |
    VERSION=$(grep -m 1 "version" Cargo.toml | cut -d '"' -f 2)
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
      "https://pypi.org/pypi/mylib/$VERSION/json")
    if [ "$HTTP_STATUS" = "200" ]; then
      echo "should_publish=false" >> $GITHUB_OUTPUT
    else
      echo "should_publish=true" >> $GITHUB_OUTPUT
    fi
```

#### Example: CI gate waiting for all checks

```yaml
ci-gate:
  needs: [version-check]
  if: needs.version-check.outputs.should_publish == 'true'
  steps:
    - uses: lewagon/wait-on-check-action@v1.3.4
      with:
        check-name: 'Rust checks'
        # ... repeat for each CI job (Python lint, tests, etc.)
```

### Lessons learned

- Lint should be a separate, fast job (no build step needed). Developers get fast feedback without waiting for compilation.
- The CI gate in the publish pipeline should wait for every check — including lint. Linted code ships; unlinted code doesn't.
- Conditional publishing (check registry → gate → build → publish) saves significant CI time vs. always building wheels.

---

## 3. Documentation

### Architecture: The Diátaxis Framework

Structure documentation around the four Diátaxis quadrants. Each quadrant serves a different user need and writing style:

```text
docs/
├── config file             ← Doc generator config (Sphinx, MkDocs, Docusaurus, etc.)
├── index                   ← Landing page with toctree organized by quadrant
│
├── Tutorials               ← Learning-oriented: "Follow along to learn"
│   └── getting-started     ← Installation + first working example
│
├── How-to Guides           ← Task-oriented: "How do I do X?"
│   ├── guides/feature-a
│   ├── guides/feature-b
│   └── ... (10-15 files)
│
├── Explanation             ← Understanding-oriented: "Why does it work this way?"
│   ├── core-concepts       ← Mental model for the library
│   ├── architecture        ← How the internals work
│   └── design-decisions    ← Why key tradeoffs were made
│
└── Reference               ← Information-oriented: "What are the exact signatures?"
    ├── api                 ← Auto-generated from source/stubs
    └── dsl-reference       ← Hand-written DSL/query language docs (if applicable)
```

**Key insight:** Most projects mix tutorials and how-to guides, or put explanatory content inside reference docs. Diátaxis forces clean separation — a tutorial walks you through learning, a how-to guide solves a specific problem, explanation helps you understand *why*, and reference gives precise details. Reorganizing existing content into these quadrants (relabeling toctree captions, moving pages) usually requires zero file renames and zero broken links.

### Key patterns

- **Auto-generated API docs:** Use your language's doc generation tools. For compiled extensions with Python bindings, `.pyi` type stubs serve as the single source of truth for both IDE autocompletion and doc generation.
- **Hosted docs with auto-deploy:** Use a service that rebuilds on every push (ReadTheDocs, GitHub Pages, Netlify). Zero manual deployment.
- **Separate guide content from README:** README is a landing page. Don't duplicate guide content there — link to the docs site instead.
- **Single source of truth for reference docs:** If you maintain reference documentation (API specs, query language docs, etc.), keep one canonical file and include/import it into both the docs site and any other surfaces.
- **Explanation pages are the most neglected quadrant.** Write at least two: one explaining *how* the internals work (architecture), one explaining *why* key decisions were made (design decisions). These pages help contributors onboard and help advanced users predict behavior.

#### Example: Sphinx AutoAPI from type stubs (Python/Rust)

For compiled extensions, you can't introspect the module at import time. Instead, write `.pyi` stubs with full docstrings and point AutoAPI at them:

```python
# docs/conf.py
extensions = ["autoapi.extension", "myst_parser"]
autoapi_dirs = ["../mylib"]
autoapi_file_patterns = ["*.pyi"]  # read stubs, not compiled modules
```

### Lessons learned

- Type stubs (`.pyi`) are underrated as a documentation tool — they provide IDE support, doc generation source, and a readable API surface all in one file.
- 5-10 task-oriented guides are more valuable than exhaustive API reference alone. Users search for "how do I do X", not "what does method Y accept."
- Include reference docs from the repo root into the docs build (via includes/symlinks). Avoids maintaining two copies that drift apart.
- Adopting Diátaxis doesn't require moving files — relabeling toctree captions and adding 2-3 explanation pages is usually enough to transform a flat doc structure into one users can navigate by intent.

---

## 4. Testing

### Structure

```text
tests/
├── conftest / setup        ← Shared fixtures + configuration
├── test_core_feature       ← Core functionality
├── test_edge_cases         ← Boundary conditions
├── test_error_handling     ← Error handling + recovery
├── test_feature_x          ← Feature-specific tests
├── test_property_based     ← Hypothesis property-based tests
└── benchmarks/             ← Performance tests (separate from CI)
    ├── test_bench_core     ← pytest-benchmark tracked benchmarks
    ├── test_performance    ← Manual timing / parametrized benchmarks
    └── test_comparison     ← vs. alternatives
```

### Key patterns

- **Tiered fixtures:** Start small, scale up. Have `empty`, `small`, `medium`, and `domain_specific` fixtures. Tests pick the smallest fixture that covers their needs. This keeps test runtime low.
- **Marker-based selection:** Tag tests with markers (`benchmark`, `slow`, `integration`). Exclude slow/benchmark tests from default CI runs. Run them on demand.
- **Strict mode:** Use `--strict-markers --strict-config` (pytest) or equivalent. Catches typos in marker names and config options that would otherwise silently be ignored.
- **Optional dependency handling:** If your library has optional features with heavy dependencies, ensure tests for those features are gracefully skipped when the dependency is missing. Use collection hooks and per-test skip guards (belt-and-suspenders).
- **Domain-specific fixtures:** If your library serves a specific domain, include realistic fixtures — they catch edge cases that synthetic data misses.
- **Coverage as informational:** Report coverage but don't block PRs on it. NumPy sets Codecov to `informational: true` — coverage metrics are useful for trend analysis, not gatekeeping. This avoids the "fix coverage to merge a typo fix" problem.
- **Property-based testing (Hypothesis):** Instead of hand-picking inputs, let Hypothesis generate random valid inputs and verify that invariants always hold. Property tests catch edge cases humans don't think to write. Good properties to test: count invariants (add N items → count == N), filter correctness (WHERE only returns matches), idempotency (creating an index doesn't change results), API parity (Cypher vs. fluent API return same data), type roundtrips (values survive store→retrieve), sort correctness, delete consistency.
- **Stubtest for compiled extensions:** If your library ships `.pyi` type stubs, run `mypy.stubtest` in CI to verify stubs match the actual runtime module. This catches real bugs: missing parameters, wrong signatures, stale methods. For PyO3/pybind11 extensions, maintain an allowlist for legitimate structural differences (no subclassing, `Option<T>` defaults, sentinel values). Stubtest found 5 real drift issues in our stubs that would have caused wrong IDE autocompletion.
- **Historical benchmark tracking:** Use pytest-benchmark to record performance metrics as JSON, then use `benchmark-action/github-action-benchmark` in CI to track results over time, alert on regressions (e.g., >150% of baseline), and auto-push results to a `gh-pages` branch. Keep existing manual-timing benchmarks for parametrized comparisons — they serve a different purpose.

#### Example: Optional dependency auto-skip (Python/pytest)

Two layers of protection — a collection hook skips entire test files, and per-test guards catch anything that slips through:

```python
# conftest.py — skip test files when optional dep is missing
import importlib.util

def pytest_collect_file(parent, file_path):
    if file_path.name.startswith("test_code_tree") and file_path.suffix == ".py":
        if importlib.util.find_spec("tree_sitter") is None:
            return None  # skip collection entirely

# test_code_tree_calls.py — per-test guard (belt-and-suspenders)
ts = pytest.importorskip("tree_sitter", reason="requires tree-sitter")
```

#### Example: Property-based tests (Hypothesis)

Test invariants that must hold for ALL valid inputs, not just hand-picked examples:

```python
from hypothesis import given, settings
from hypothesis import strategies as st

@given(n=st.integers(min_value=1, max_value=50))
@settings(max_examples=30)
def test_node_count_invariant(n):
    """Adding N nodes increases count by exactly N."""
    graph = KnowledgeGraph()
    df = pd.DataFrame({"nid": list(range(n)), "name": [f"item_{i}" for i in range(n)]})
    graph.add_nodes(df, "Thing", "nid", "name")
    result = graph.select("Thing").collect()
    assert len(result) == n

@given(n=st.integers(min_value=5, max_value=30),
       target_city=st.sampled_from(["Oslo", "Bergen", "Stavanger"]))
@settings(max_examples=30)
def test_filter_returns_only_matching(n, target_city):
    """WHERE filter only returns nodes matching the condition."""
    # ... build graph with city property ...
    result = graph.select("Person").where({"city": target_city}).collect()
    for node in result:
        assert node["city"] == target_city
```

#### Example: Stubtest for compiled extensions

```bash
# Run stubtest to verify .pyi stubs match the compiled module
python -m mypy.stubtest mylib --allowlist stubtest_allowlist.txt
```

The allowlist handles legitimate PyO3/pybind11 structural differences:

```text
# stubtest_allowlist.txt

# PyO3 classes cannot be subclassed (@final) and are disjoint bases.
mylib.MyClass

# PyO3 uses Option<T>, showing default=None at runtime.
# Stubs document effective defaults (e.g., top_k=10).
mylib.MyClass.method_with_option_defaults

# Overloaded methods structurally differ from runtime's single dispatch.
mylib.MyClass.overloaded_method
```

CI step (run after building the extension, one Python version only):

```yaml
- name: Run stubtest
  if: matrix.python-version == '3.12'
  run: |
    pip install mypy
    python -m mypy.stubtest mylib --allowlist stubtest_allowlist.txt
```

#### Example: Historical benchmark tracking (pytest-benchmark + CI)

```python
@pytest.mark.benchmark
def test_bench_add_nodes(benchmark):
    """Bulk node insertion (1000 nodes)."""
    graph = MyLib()
    data = build_test_data(1000)
    benchmark(graph.add_nodes, data)
```

Makefile targets for local use:

```makefile
bench-save:
	pytest tests/benchmarks/ -m benchmark --benchmark-save=baseline
bench-compare:
	pytest tests/benchmarks/ -m benchmark --benchmark-compare
```

CI tracking on pushes to main:

```yaml
- name: Run tracked benchmarks
  run: pytest tests/benchmarks/ -m benchmark --benchmark-json=benchmark-results.json

- name: Store benchmark result
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'pytest'
    output-file-path: benchmark-results.json
    auto-push: true
    alert-threshold: '150%'
    comment-on-alert: true
    fail-on-alert: false      # alert but don't block — benchmarks are noisy
```

#### Example: Codecov config (informational, not blocking)

```yaml
# .codecov.yml
coverage:
  status:
    project:
      informational: true    # report but don't fail
    patch:
      informational: true
comment: off                  # don't clutter PR comments
```

#### Example: Coverage config (Python)

```toml
[tool.coverage.run]
source = ["mylib"]
omit = ["mylib/optional_module/*"]    # exclude optional-dep modules

[tool.coverage.report]
exclude_lines = ["if __name__", "pragma: no cover", "raise NotImplementedError"]
```

### Lessons learned

- Comparison benchmarks against alternatives (e.g., your library vs. the incumbent) are valuable for both development decisions and marketing.
- The `pytest_collect_file()` hook is cleaner than per-file skip logic for handling optional dependencies across many test files.
- A `make cov` target that shows missing lines locally is more actionable than a badge showing a percentage.
- Strict marker/config validation catches silent test misconfiguration early.
- Property-based tests find bugs that hand-written tests miss. The investment is ~30 lines per property, and Hypothesis shrinks failing cases to minimal reproducers automatically.
- Stubtest is the single most effective tool for keeping type stubs in sync with a compiled extension. It found 5 real signature drift issues that manual review missed — wrong parameter names, missing parameters, stale overloads.
- For PyO3 extensions, stubtest requires a well-maintained allowlist. Group allowlist entries by category (structural differences, Option defaults, overloads) with comments explaining why each entry exists.
- Historical benchmark tracking with `fail-on-alert: false` is the right default. Benchmarks are inherently noisy — alerting on regressions is useful, but blocking merges on them creates false negatives. Use alerts for investigation, not gates.

---

## 5. Project Metadata

### Essential fields

Every package registry has metadata fields that affect discoverability. Fill them all.

```text
name, version, description, keywords, classifiers/categories,
homepage, repository, documentation, changelog, issues
```

### Key patterns

- **Single version source of truth:** Define the version in one place (`Cargo.toml`, `package.json`, `pyproject.toml`, etc.). Never duplicate it. Build tools should read from this single source.
- **Keywords:** Registries index these for search. Use domain-specific terms, not language names.
- **Project URLs:** Most registries show these prominently. Include Documentation, Changelog, and Issues at minimum — they signal a well-maintained project.
- **Optional dependencies:** Heavy or niche dependencies should be extras/optional features, not core requirements. Keep the core install lightweight. Provide clear runtime error messages when an optional dependency is missing.
- **PEP 561 `py.typed` marker (Python):** If your package ships type stubs (`.pyi` files), include an empty `py.typed` file in the package root. This tells type checkers (mypy, pyright) to use your inline types/stubs. Without it, your stubs are invisible to type checkers. NumPy ships this.

#### Example: Optional dependency with runtime guard (Python)

Move heavy deps to extras in `pyproject.toml`:

```toml
dependencies = ["pandas>=1.5"]          # lightweight core

[project.optional-dependencies]
code-tree = [                           # heavy optional feature
    "tree-sitter>=0.21",
    "tree-sitter-rust",
    "tree-sitter-python",
    # ... more language parsers
]
```

Then guard the module at runtime:

```python
# mylib/code_tree/__init__.py
try:
    import tree_sitter
except ImportError:
    raise ImportError(
        "The code_tree module requires tree-sitter. "
        "Install with: pip install mylib[code-tree]"
    ) from None
```

### Lessons learned

- Adding Changelog and Issues URLs to package metadata provides immediate value — registry UIs show these as clickable sidebar links.
- Moving heavy optional dependencies out of core requirements can significantly reduce install time and size for most users.
- Runtime guards with helpful error messages ("Install with `pip install mylib[feature]`") are better than cryptic import errors.

---

## 6. Linting & Formatting

### The local-first principle

**Every CI check must be reproducible locally with a single command.** If `make lint` passes on your machine, CI will pass. No surprises after pushing.

This is the most important principle in this entire document. It means:

1. **CI and local lint run the exact same checks** — same tools, same flags, same config file. Don't add CI-only checks that developers can't reproduce locally.
2. **One command to verify** — `make lint` (or equivalent) runs all checks. One command, one answer: pass or fail.
3. **One command to fix** — `make fmt` (or equivalent) auto-fixes everything. The fix → verify loop should take seconds, not minutes.
4. **CI gate enforces lint before publish** — the publish pipeline waits for lint to pass, not just tests. Unlinted code never ships.

### Adding linting to an existing codebase

When retrofitting linting onto a project with existing code:

1. **Start conservative** — use a wide line-length and minimal rule set. Match your existing style, don't fight it.
2. **Baseline in one commit** — run auto-format + auto-fix to clean up everything at once. This is a single formatting commit.
3. **Fix remaining issues manually** — auto-fix handles ~90%. The rest are real issues (unused variables, ambiguous names) or false positives.
4. **Handle false positives with targeted suppression** — use inline suppression comments on specific lines. Use per-file ignore patterns for known patterns. Never disable rules globally to silence a few false positives.
5. **Document why rules are disabled** — NumPy comments every disabled rule in their `ruff.toml`. Future maintainers need to know if a rule was disabled intentionally or lazily.
6. **Enforce in CI immediately** — once the baseline commit is clean, add the CI check. Don't delay — the window between "we have a linter" and "CI enforces it" is when style drift creeps back in.

#### Example: Ruff config for Python

```toml
[tool.ruff]
line-length = 120              # match existing style, avoid mass reformatting
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]  # pycodestyle, pyflakes, isort, warnings

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["E402"]           # allow imports after pytest.importorskip()
"benchmarks/*" = ["E402"]
```

#### Example: Handling false positives

Some patterns trigger legitimate-looking lint errors that are actually correct code. Suppress with targeted inline comments, not global rule disabling:

```python
# DuckDB reads Python locals by name — the variable IS used, just not by Python
df = pd.DataFrame(data)  # noqa: F841
db.execute("CREATE TABLE t AS SELECT * FROM df")
```

#### Example: Makefile targets (Rust + Python project)

```makefile
fmt:                                          # format all code
	cargo fmt
fmt-py:                                       # format + auto-fix Python
	ruff format . && ruff check --fix .
lint:                                         # ALL checks — must match CI
	cargo fmt -- --check
	cargo clippy -- -D warnings
	ruff format --check . && ruff check .
lint-py:                                      # Python lint only
	ruff format --check . && ruff check .
```

For single-language projects, simplify to just `fmt` and `lint`.

### Lessons learned

- Lint should be a separate CI job that doesn't require building the project. Fast feedback.
- Conservative initial config avoids a 500-file reformatting PR that obscures git blame. Start with rules that match your style, tighten later.
- Per-file ignore patterns (not global rule disabling) are the right way to handle false positives from test framework patterns.
- The convention "always run `make lint` before pushing" should be documented in your contributor guide. It's the single most effective way to avoid CI failures.
- Document every disabled rule with a comment explaining why (learned from NumPy's `ruff.toml`).

---

## 7. Developer Experience

### Contributor documentation

Every project should have a conventions file (CONTRIBUTING.md, CLAUDE.md, DEVELOPMENT.md, or similar) that covers:

1. **Build commands** — exact steps to build, test, and lint
2. **Architecture overview** — where things live, key patterns, module responsibilities
3. **Checklist for changes** — which files to update when modifying the public API
4. **Commit conventions** — format, changelog policy
5. **Release process** — version bumping, who approves pushes

NumPy keeps the root `CONTRIBUTING.rst` minimal (just a welcome message and link to full docs). Comprehensive contributor documentation lives in the docs site, not in a massive root file.

#### Example: API change checklist

When changing a public method, update all of these:

1. Implementation source code
2. Type stubs / interface definitions (`.pyi`, `.d.ts`, etc.)
3. Schema introspection output (if your library self-describes for agents/tools)
4. Integration entry points (MCP server, CLI, REST API)
5. Changelog (`[Unreleased]` section)

This prevents the most common contributor mistake: changing the implementation but forgetting to update docs, stubs, or changelog.

### Task runner

A task runner with common targets saves developers from memorizing commands. Keep it simple — one line per target.

Recommended targets:

```makefile
dev       # Build and install locally
test      # Run all tests
lint      # Run all lint checks — must match CI exactly
fmt       # Auto-format all code
cov       # Run tests with coverage report
bench     # Run benchmarks
clean     # Remove build artifacts
```

Works with Make, Just, Task, npm scripts, Cargo aliases, spin (NumPy uses this), or any other task runner.

### `.editorconfig`

A one-time, 10-line file that enforces consistent formatting across all editors (VS Code, JetBrains, vim, Emacs) without requiring project-specific plugins. Prevents entire classes of whitespace noise in diffs.

```ini
# .editorconfig
root = true

[*]
indent_style = space
indent_size = 4
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true

[*.md]
trim_trailing_whitespace = false

[Makefile]
indent_style = tab
```

### Examples

Include 3-5 runnable example scripts:

- One minimal "hello world" example
- One domain-specific example showing realistic usage
- One integration example (e.g., server, API endpoint, notebook)

### Lessons learned

- A Makefile (or equivalent) is the single best DX investment. New contributors can be productive in minutes.
- The API change checklist prevents sync issues between implementation and public surface. Without it, stubs/docs/changelog drift is inevitable.
- Examples serve double duty: they're documentation for users and smoke tests for maintainers.
- `.editorconfig` is a one-time investment that prevents recurring whitespace issues across all contributors.

---

## 8. Community & GitHub Scaffolding

Mature projects invest in scaffolding that guides contributors, automates triage, and reduces maintainer burden. These patterns are free to set up and pay off immediately.

### Issue templates (YAML forms)

Use GitHub's YAML-based issue forms instead of Markdown templates. They provide structured input fields, required fields, and dropdowns — resulting in higher-quality reports with less back-and-forth.

Recommended templates:

- **Bug report** — reproduction steps, expected vs. actual, environment info (required fields)
- **Feature request** — motivation, proposed API, alternatives considered
- **config.yml** — optionally redirect questions to Discussions or allow blank issues

```yaml
# .github/ISSUE_TEMPLATE/bug-report.yml
name: Bug Report
description: Report a bug
body:
  - type: textarea
    id: description
    attributes:
      label: Describe the bug
    validations:
      required: true
  - type: textarea
    id: reproduce
    attributes:
      label: Steps to reproduce
      description: Minimal code to reproduce the issue
      render: python
    validations:
      required: true
  - type: input
    id: version
    attributes:
      label: Library version
    validations:
      required: true
```

```yaml
# .github/ISSUE_TEMPLATE/config.yml
blank_issues_enabled: true    # or false + contact_links to redirect to Discussions
```

### Pull request template

Keep it minimal. Link to your workflow docs; don't duplicate them.

```markdown
<!-- .github/PULL_REQUEST_TEMPLATE.md -->
## Summary

<!-- Brief description of what this PR does -->

## Checklist

- [ ] Tests pass (`make test`)
- [ ] Lint passes (`make lint`)
- [ ] Changelog updated (if user-visible)
- [ ] Docs updated (if API changed)
```

### Dependabot

Automate dependency updates. Most vulnerabilities come from stale dependencies.

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    commit-message:
      prefix: "chore"
```

### Security policy

Add `SECURITY.md` to `.github/` with instructions for reporting vulnerabilities privately. GitHub surfaces this in the "Security" tab. Even small projects benefit — it shows you've thought about security.

### Lessons learned (from NumPy)

- YAML issue forms produce dramatically better bug reports than free-form Markdown templates. Required fields eliminate the "can you provide a reproducer?" round-trip.
- Coverage should inform, not block. Setting Codecov to `informational: true` avoids the "fix coverage to merge this typo fix" problem.
- `.editorconfig` is a one-time 10-line file that prevents entire classes of formatting noise in diffs.
- Dependabot for GitHub Actions is especially valuable — action versions frequently have security patches.
- Document *why* each disabled lint rule is disabled (NumPy comments every rule in `ruff.toml`). Future maintainers need to know if a rule was disabled intentionally or lazily.

---

## 9. Versioning & Releases

### Recommended approach

- **Version source of truth:** One file, one place. Build tools, CI, and docs all read from it.
- **Changelog format:** [Keep a Changelog](https://keepachangelog.com/) with sections: Added, Changed, Fixed, Deprecated, Removed.
- **Commit format:** `type: short description` (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`).
- **Release flow:** Bump version → promote `[Unreleased]` in changelog → commit → push → CI publishes automatically.
- **Automated publishing:** CI checks if version exists in registry. If not, builds artifacts and publishes. No manual uploads.

### Lessons learned

- Automated conditional publishing (check registry → gate → build → publish) eliminates the "did someone remember to publish?" question.
- Extracting changelog notes into GitHub/GitLab releases automatically ensures every release has documentation.
- Frequent small releases (even multiple per day during active development) are better than infrequent large ones. Automated publishing makes this painless.

---

## 10. Checklist for New Projects

```text
□ Repository
  □ Description (keyword-rich, one sentence)
  □ Topics / tags (6-10 domain-specific)
  □ Social preview image (1280×640px)
  □ License file
  □ .gitignore
  □ .editorconfig

□ README
  □ Title with keywords
  □ Badges (package registry, docs, license, CI, coverage)
  □ Keyword-rich opening paragraph
  □ "Why X?" section
  □ Quick start with install command + minimal example
  □ Use cases with code snippets
  □ Key features table
  □ Top 5 doc links
  □ Call for contributions (welcome non-code contributions)
  □ Requirements + license at bottom

□ CI/CD
  □ Lint as separate fast job (no build required)
  □ Tests with version matrix + strict mode
  □ Coverage collection + upload (informational, not blocking)
  □ Conditional publish (version check → CI gate → build → publish)
  □ CI gate waits for ALL checks before publish
  □ Platform matrix (if compiled / native code)
  □ Keyless / trusted publishing
  □ Benchmark workflow (manual trigger)

□ Documentation (Diátaxis framework)
  □ Doc generator with auto-deploy on push
  □ Tutorials: getting started guide
  □ How-to Guides: 5-10 task-oriented guides
  □ Explanation: architecture + design decisions pages
  □ Reference: auto-generated API + hand-written DSL docs
  □ Single source of truth for reference docs

□ Testing
  □ Tiered fixtures (empty → small → medium → domain)
  □ Optional dependency auto-skip
  □ Benchmark / slow test separation via markers
  □ Coverage reporting (local + CI)
  □ Strict marker/config validation
  □ Property-based tests (Hypothesis) for core invariants
  □ Stubtest verification (.pyi stubs match compiled extension)
  □ Historical benchmark tracking (pytest-benchmark + CI action)

□ Project Metadata
  □ Keywords and categories in package manifest
  □ Project URLs (docs, changelog, issues)
  □ Single version source of truth
  □ Optional dependency groups for heavy deps
  □ Lightweight core install
  □ PEP 561 py.typed marker (if shipping type stubs)

□ Linting (local-first)
  □ Auto-formatter configured
  □ Linter configured
  □ `make lint` runs ALL checks (must match CI)
  □ `make fmt` auto-fixes everything
  □ Lint gate in publish pipeline
  □ Targeted suppression for false positives
  □ Disabled rules documented with comments

□ Developer Experience
  □ Contributor guide with build/test/lint commands
  □ Task runner with common targets
  □ API change checklist
  □ 3-5 runnable examples
  □ Convention: "run make lint before pushing"

□ Community & GitHub Scaffolding
  □ Issue templates (YAML forms: bug, feature)
  □ Pull request template with checklist
  □ Dependabot for actions + dependencies
  □ SECURITY.md for vulnerability reporting
```
