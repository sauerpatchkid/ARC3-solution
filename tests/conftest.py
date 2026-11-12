"""conftest.py - the import paths every test needs, set once.

pytest loads this file before any test in this folder. The repo is not an
installed package: live code is imported by bare module name, the way
run_local.py and the tools import it:

    from the repo root        manifest, tiers, registry, compute_metrics, ...
    from custom_agents/       canon, return_map, upgrades, gb.s5_act, ...
    from tools/               gb_gate, gb_sweep, paired_compare, ...
    from tests/               helpers shared between test files (test_gb_core.play, ...)

Until 10 Nov 2026 every test file repeated these sys.path lines itself.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "custom_agents"), os.path.join(ROOT, "tools"),
              os.path.join(ROOT, "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
