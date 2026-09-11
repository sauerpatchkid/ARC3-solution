#!/usr/bin/env python3
"""check_repo.py — fast health check. Run this after cloning, and before you push.

    uv run python check_repo.py        (or: make check)

Four checks, each of which has caught a real problem in this repo:

  1. ISOLATION — no baseline module may import llm_track.
     This is the guarantee that Matt's LLM work cannot change anyone else's
     numbers. llm_track/ may import from the baseline (it reads the corpus and
     the canonicalizer); the arrow must never point the other way, because then
     an LLM-side edit would silently alter a teammate's baseline run. Baseline
     tooling (the root Makefile, sweep.sh) must not invoke it or .venv-llm
     either: its commands live in llm_track/Makefile.

  2. REGISTRY — every agent in custom_agents/__init__.py actually imports and
     has the surface run_local.py drives. A typo in REGISTRY otherwise only
     shows up hours into a sweep.

  3. DEPENDENCIES — every third-party module the baseline imports is installed.
     requirements.txt has been incomplete before (xxhash and Pillow were both
     missing while the agent imported them), which breaks a fresh clone.

  4. BENCHMARK — the frozen suites parse and name real games.

Exit status is non-zero if anything fails, so it works in CI or a git hook.
"""
import ast
import importlib.util
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Everything a teammate's baseline run touches. llm_track/ is deliberately NOT
# here: it is the isolated side.
BASELINE_FILES = [
    "run_local.py", "run_curriculum.py", "compute_metrics.py",
    "analyze_curves.py", "metrics_common.py", "eval_common.py",
    "summarize_overnight.py", "inspect_corpus.py", "utils.py",
    "benchmark.py", "compare.py", "custom_agent.py",
    "custom_agents/__init__.py", "custom_agents/action.py",
    "custom_agents/random_agent.py", "custom_agents/view_utils.py",
    "custom_agents/TEMPLATE.py",
]

# Modules that only exist once run_local.py has synthesized the harness package,
# or that come from the engine/harness rather than pip.
PROVIDED_AT_RUNTIME = {"agents", "arc_agi", "arcengine"}
# First-party modules in this repo.
LOCAL = {"eval_common", "metrics_common", "utils", "view_utils", "action",
         "random_agent", "custom_agents", "custom_agent", "benchmark",
         "compare", "run_local", "compute_metrics", "summarize_overnight",
         "analyze_curves", "inspect_corpus", "llm_track"}

# Baseline tooling that must never run LLM code or use its environment.
BASELINE_TOOLING = ["Makefile", "sweep.sh"]
LLM_INVOCATION = re.compile(r"-m\s+llm_track|\.venv-llm|-C\s+llm_track")

failures = []
notes = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def imported_modules(path):
    """Top-level module names imported by a file (static parse, no execution)."""
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.add(node.module.split(".")[0])
    return out


def check_isolation():
    print("\n1. Isolation: baseline code and tooling never touch llm_track")
    bad = []
    for rel in BASELINE_FILES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            notes.append(f"{rel} not found (skipped)")
            continue
        if "llm_track" in imported_modules(p):
            bad.append(rel)
    check("baseline does not depend on llm_track", not bad,
          f"offenders: {', '.join(bad)}" if bad else
          f"{len(BASELINE_FILES)} files clean")
    bad_tool = [rel for rel in BASELINE_TOOLING
                if os.path.exists(os.path.join(ROOT, rel))
                and LLM_INVOCATION.search(open(os.path.join(ROOT, rel)).read())]
    check("baseline tooling does not run llm_track or .venv-llm", not bad_tool,
          f"offenders: {', '.join(bad_tool)}" if bad_tool else
          f"{', '.join(BASELINE_TOOLING)} clean")


def check_registry():
    print("\n2. Agent registry")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
    from custom_agents import REGISTRY

    # Agent modules import `agents.structs`, which run_local.py synthesizes.
    # Do the same so we can import them here.
    import run_local
    run_local._install_minimal_agents_pkg()

    required = ("is_done", "choose_action")
    for name, (module_name, class_name) in sorted(REGISTRY.items()):
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            miss = [m for m in required if not hasattr(cls, m)]
            check(f"agent {name!r} ({module_name}.{class_name})", not miss,
                  f"missing {', '.join(miss)}" if miss else "imports, has the surface")
        except Exception as e:
            check(f"agent {name!r} ({module_name}.{class_name})", False,
                  f"{type(e).__name__}: {e}")


def check_dependencies():
    print("\n3. Dependencies importable")
    needed = set()
    for rel in BASELINE_FILES:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            needed |= imported_modules(p)
    third_party = sorted(m for m in needed
                         if m not in LOCAL
                         and m not in PROVIDED_AT_RUNTIME
                         and m not in sys.stdlib_module_names)
    missing = [m for m in third_party if importlib.util.find_spec(m) is None]
    check("third-party imports installed", not missing,
          f"MISSING: {', '.join(missing)} — add to requirements.txt"
          if missing else f"{', '.join(third_party)}")
    for m in PROVIDED_AT_RUNTIME - {"agents"}:
        if importlib.util.find_spec(m) is None:
            notes.append(f"{m} not installed — the local engine will not run "
                         f"(see README: pip install arc-agi)")


def check_benchmark():
    print("\n4. Benchmark suites")
    sys.path.insert(0, ROOT)
    import benchmark
    known = set(benchmark.ALL_GAMES)
    for name, s in benchmark.SUITES.items():
        unknown = [g for g in s.games if g not in known]
        check(f"suite {name!r}", not unknown and bool(s.games) and bool(s.seeds),
              f"unknown games {unknown}" if unknown else
              f"{s.n_runs} runs, ~{s.est_hours:.1f} h")


def main():
    print("check_repo.py — repository health")
    check_isolation()
    check_registry()
    check_dependencies()
    check_benchmark()
    if notes:
        print("\nNotes:")
        for n in notes:
            print(f"  - {n}")
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
