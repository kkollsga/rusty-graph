# Example: code review with `workspace.kind: local` + `watch`

Bind a local source directory as the active source root for source
tools (`read_source` / `grep` / `list_source`), build a code-tree
graph over it, and auto-rebuild on file changes. The agent can swap
between sibling project directories with `set_root_dir(path)`
without restarting the server.

## Manifest

```yaml
# code_review_mcp.yaml — under /Volumes/EksternalHome/Koding/
name: Code Review
instructions: |
  Code review against any project under Koding/. The active root
  starts at the directory below; swap with set_root_dir(path).

workspace:
  kind: local
  root: /Volumes/EksternalHome/Koding   # the sandbox boundary
  watch: true                            # auto-rebuild on file changes
```

## What gets registered

`tools/list` returns:

```
- cypher_query
- graph_overview
- read_code_source
- ping
- read_source / grep / list_source
- repo_management              # local-mode also gets this for rebuild dispatch
- set_root_dir                 # local-mode only
- github_issues / github_api   # if GITHUB_TOKEN is in env
```

## The flow

1. Boot: the server canonicalises `/Volumes/.../Koding/` and binds
   it as the active root. `read_source` / `grep` / `list_source`
   sandbox to it.
2. The watcher starts on the root with a 500 ms debounce.
3. First agent action: `cypher_query` against the auto-built
   code-tree graph (modules, functions, calls, imports).
4. Agent narrows interest to a specific file → calls `read_source`.
5. Agent decides to switch to a different project under Koding/:
   `set_root_dir("/Volumes/EksternalHome/Koding/Rust/KGLite")`.
   The active root atomically swaps; source tools rebind; the watch
   handle moves to the new root.
6. On any file change inside the new root, the watcher fires (after
   the 500 ms debounce window), the code-tree rebuilds on a
   background thread, and the new graph atomically swaps in. Queries
   against the previous graph keep working until the swap.

## Sandbox boundary

`workspace.root` is the manifest-declared root. The sandbox check on
`set_root_dir(path)` validates against this immutable boundary, not
the current active root. After `set_root_dir(/.../Rust/KGLite)`,
a subsequent `set_root_dir(/.../MCP servers)` works because both
paths are under `workspace.root`:

```
{"name":"set_root_dir","arguments":{"path":"/Volumes/EksternalHome/Koding/Rust/KGLite"}}
→ "Active root set to /Volumes/EksternalHome/Koding/Rust/KGLite."   ✓

{"name":"set_root_dir","arguments":{"path":"/Volumes/EksternalHome/Koding/MCP servers"}}
→ "Active root set to /Volumes/EksternalHome/Koding/MCP servers."   ✓ (sibling swap, no restart)

{"name":"set_root_dir","arguments":{"path":"/tmp"}}
→ "Error: path '/tmp' escapes the workspace root."                 ✓ (outside the sandbox)
```

## `repo_management` in local mode

In local-workspace mode `repo_management` does not accept a repo
name (there's nothing to clone). It accepts:

```text
{"update": true}                        — re-fingerprint + rebuild
{"update": true, "force_rebuild": true} — rebuild regardless of fingerprint
```

`force_rebuild: true` bypasses the SHA gating that's otherwise on
top of the post-activate hook — useful after upgrading kglite
itself or after a code-tree-builder change.

## github_issues integration

If the project under the active root has a `.git/config` with a
GitHub remote, `github_issues` auto-resolves the `repo_name`
argument from the active root's git config. Agents can call

```json
{"name": "github_issues", "arguments": {"limit": 5, "state": "all"}}
```

without supplying `repo_name`; the active workspace repo is used as
the fallback. The token still has to be in env / walked-up `.env`.

## Performance notes

- Code-tree rebuild is incremental for small files (a single-file
  edit is fast). Whole-tree rebuilds on large projects (>100k LoC)
  can take a few seconds — the debounce + background thread keep
  the agent unblocked during the rebuild.
- Atomic swap means agents never see a half-built graph. Queries
  against the in-flight graph complete; the next query after the
  swap sees the new graph.
- The `.mcp-workspace/` directory inside `workspace.root` stores
  inventory + last-built SHA. Safe to delete — the next boot
  re-seeds it.
