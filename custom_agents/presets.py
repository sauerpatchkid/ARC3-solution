"""presets.py — named configurations of Goose, so a comparison names ONE thing.

The adopted agent is three EVAL_* settings on top of the default Goose. Writing
those three out again in every script and doc is how two "identical" arms end
up different; here they are written once and given the name the results docs
already use.

    uv run python run_local.py --game ft09 --agent mb_gated_att
    make bench SUITE=standard AGENT=mb_gated_att

`goose` (no flags) is unchanged: it stays the semester-1 baseline teammates
compare against. A preset FIXES its settings: if the environment asks for a
different value of one of them the run stops, rather than quietly being some
other agent under this name. Settings the preset does not name (EVAL_SEED,
EVAL_MAX_ACTIONS, EVAL_MAP_STALL, ...) pass through as usual.

A name here means the same thing forever. When a new configuration is adopted,
add a new preset; do not edit an old one (same rule as benchmark.py's suites).
"""
import os

from action import Action

# Adopted 2026-09-28: 112 vs 79 levels on 25 games x 3 seeds
# (docs/plans/upgrade-confirm-results.md).
MB_GATED_ATT = {"EVAL_LABEL": "novel", "EVAL_RETURN_MAP": "1",
                "EVAL_UPGRADES": "bars,map_gated,attempt"}


def _same(a, b):
    """Equal as settings: case, spacing and the order of a comma list do not matter."""
    norm = lambda v: sorted(x.strip().lower() for x in str(v).split(",") if x.strip())
    return norm(a) == norm(b)


def apply(settings, name):
    """Put a preset's settings into the environment, refusing a conflicting one."""
    for key, value in settings.items():
        asked = os.environ.get(key)
        if asked is not None and not _same(asked, value):
            raise SystemExit(f"--agent {name} fixes {key}={value}, but the environment has "
                             f"{key}={asked}. Unset it, or use --agent goose with your own flags.")
        os.environ[key] = value


class MbGatedAtt(Action):
    """The adopted Goose: novelty label + return map + bars, map_gated, attempt."""

    def __init__(self, *args, **kwargs):
        apply(MB_GATED_ATT, "mb_gated_att")
        super().__init__(*args, **kwargs)
