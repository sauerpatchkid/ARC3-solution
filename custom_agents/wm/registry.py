"""registry.py - the list of frozen artifacts (docs/plans/rulebook-v2.md sections
2.4 and 3.3): `artifacts/registry.json`, tracked in git.

An entry says: this set of files, with these hashes, was frozen on this date and
was built from these games. Two kinds of thing go in:
  * cross-game artifacts (helper library, schema catalog, playbook, Goose prior,
    a fine-tuned writer) - frozen and registered BEFORE any held-out use;
  * frozen result sets that later work depends on (v1's rule candidates).

    uv run python tools/artifacts.py add v1_results results/rulebook/tier0a ... --games dev,seen
    uv run python tools/artifacts.py verify          # every file still has its hash

An entry is never edited: a changed artifact is a new entry with a new name.
The files themselves may live in results/ (not in git); the hashes are what make
a later change visible.
"""
import hashlib
import json
import os
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(ROOT, "artifacts", "registry.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files(paths, root=ROOT):
    out = []
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(root, p)
        if os.path.isdir(full):
            for d, _, names in sorted(os.walk(full)):
                out += [os.path.join(d, n) for n in sorted(names)]
        else:
            out.append(full)
    return sorted(set(out))


def load(path=PATH):
    if not os.path.exists(path):
        return {"artifacts": [], "events": []}
    with open(path) as f:
        return json.load(f)


def save(reg, path=PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(reg, f, indent=1)
        f.write("\n")


def add(name, paths, source_games, note="", kind="artifact", path=PATH, root=ROOT, today=None):
    """Freeze `paths` (files or folders) under `name`. Refuses a name already used."""
    reg = load(path)
    if any(a["name"] == name for a in reg["artifacts"]):
        raise SystemExit(f"artifact {name!r} is already registered; a changed artifact needs a new name")
    files = {os.path.relpath(f, root): sha256(f) for f in _files(paths, root)}
    if not files:
        raise SystemExit(f"nothing to register under {paths}")
    whole = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    entry = {"name": name, "kind": kind, "frozen": today or date.today().isoformat(),
             "source_games": sorted(source_games), "note": note, "n_files": len(files),
             "sha256": whole, "files": files}
    reg["artifacts"].append(entry)
    save(reg, path)
    return entry


def verify(name=None, path=PATH, root=ROOT):
    """[(artifact, file, problem)] for every registered file that is missing or changed."""
    bad = []
    for a in load(path)["artifacts"]:
        if name and a["name"] != name:
            continue
        for rel, want in a["files"].items():
            full = os.path.join(root, rel)
            if not os.path.exists(full):
                bad.append((a["name"], rel, "missing"))
            elif sha256(full) != want:
                bad.append((a["name"], rel, "changed"))
    return bad


def log_event(what, detail, path=PATH, today=None):
    """Append a dated line to the registry's event log (for example: sealed
    results opened, and why)."""
    reg = load(path)
    reg.setdefault("events", []).append({"date": today or date.today().isoformat(),
                                         "what": what, "detail": detail})
    save(reg, path)
