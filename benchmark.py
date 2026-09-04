#!/usr/bin/env python3
"""benchmark.py — THE frozen test set. One definition, everyone runs it.

This file is the single source of truth for what "the benchmark" means. The
sweep runner, the Makefile and the comparison table all read the suites from
here, so two people who run

    make bench SUITE=standard AGENT=goose
    make bench SUITE=standard AGENT=theirs

have provably played the same games, with the same seeds, for the same number
of actions, and their numbers can be put side by side.

CHANGING A SUITE INVALIDATES EVERY COMPARISON ALREADY MADE WITH IT. If you need
a different set of games or budget, add a NEW suite rather than editing one —
the whole point is that `standard` means the same thing in November as it did
in September. Suites are versioned; bump the version if you must change one.

    uv run python benchmark.py                    # list the suites
    uv run python benchmark.py --suite standard   # show one in detail
    make bench SUITE=standard AGENT=goose         # run it

WHY THESE GAMES (suite `standard`)
  Picked from the 25-game sweep (results/sweeps/sweep_20260727_225828_summary.md),
  which found 12 of 25 games complete at least one level:
    ft09, ar25, cd82, lp85 — the four games that reach level 2, so an agent has
                             room to show an improvement in level progress.
    ls20                   — the null contrast: it completes nothing, so any
                             "improvement" reported here is a measurement
                             artefact, not a result.
    dc22                   — mid-novelty, completes nothing, and has semester-1
                             medians for continuity with earlier numbers.

WHY THREE SEEDS
  Runs are NOT reproducible past the first training step: identical code and
  EVAL_SEED diverge once nondeterministic CUDA kernels touch the weights
  (measured: identical for ~300 actions, level-up at 1110 vs 1423 on the same
  seed). A single seed is therefore a sample, not a measurement. Three is the
  minimum for a median; report "k/n seeds reached level k" alongside it, never
  a bare median. See "Known measurement caveats" in CLAUDE.md.
"""
import argparse

# Rough per-action throughput used only for the runtime estimate printed below.
ACT_PER_SEC = 130

# The 25 public games, for the `full` suite.
ALL_GAMES = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
]


class Suite:
    """A frozen (games, seeds, cap) triple with a name and a reason."""

    def __init__(self, name, version, games, seeds, cap, purpose):
        self.name = name
        self.version = version
        self.games = tuple(games)
        self.seeds = tuple(seeds)
        self.cap = cap
        self.purpose = purpose

    @property
    def n_runs(self):
        return len(self.games) * len(self.seeds)

    @property
    def est_hours(self):
        return self.n_runs * self.cap / ACT_PER_SEC / 3600

    def describe(self):
        return (f"{self.name} (v{self.version}): {len(self.games)} games x "
                f"{len(self.seeds)} seeds x {self.cap:,} actions = "
                f"{self.n_runs} runs, ~{self.est_hours:.1f} h\n"
                f"    games: {' '.join(self.games)}\n"
                f"    seeds: {' '.join(str(s) for s in self.seeds)}\n"
                f"    {self.purpose}")


SUITES = {
    "smoke": Suite(
        "smoke", 1, ["ft09"], [0], 2000,
        "30-second sanity check: does this agent run and produce a corpus?"),

    "quick": Suite(
        "quick", 1, ["ft09", "ls20", "cd82"], [0, 1], 20000,
        "~15 min. For iterating. NOT for reporting: 20k actions is below the "
        "budget at which most games complete a second level."),

    "standard": Suite(
        "standard", 1, ["ft09", "ar25", "cd82", "lp85", "ls20", "dc22"],
        [0, 1, 2], 100000,
        "THE COMPARISON SUITE. Report this one. See the header for why these "
        "games and why three seeds."),

    "full": Suite(
        "full", 1, ALL_GAMES, [0], 100000,
        "All 25 public games, 1 seed. Coverage breadth, not statistical "
        "power - single-seed numbers are samples; do not read differences "
        "between agents on one game as real."),
}

DEFAULT_SUITE = "standard"


def get(name):
    if name not in SUITES:
        raise SystemExit(f"unknown suite {name!r}. Available: "
                         f"{', '.join(SUITES)}")
    return SUITES[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default=None, help="show one suite in detail")
    ap.add_argument("--env", action="store_true",
                    help="emit shell assignments for sweep.sh (internal)")
    a = ap.parse_args()

    if a.env:
        s = get(a.suite or DEFAULT_SUITE)
        # Consumed by sweep.sh via eval; keep the format stable.
        print(f'GAMES="{" ".join(s.games)}"')
        print(f'SEEDS="{" ".join(str(x) for x in s.seeds)}"')
        print(f'CAP={s.cap}')
        print(f'SUITE_VERSION={s.version}')
        return

    if a.suite:
        print(get(a.suite).describe())
        return

    print("Frozen benchmark suites (benchmark.py)\n")
    for s in SUITES.values():
        print("  " + s.describe().replace("\n", "\n  "))
        print()
    print(f"Run one:  make bench SUITE={DEFAULT_SUITE} AGENT=goose")
    print("Compare:  make compare M1=<manifest> M2=<manifest>")


if __name__ == "__main__":
    main()
