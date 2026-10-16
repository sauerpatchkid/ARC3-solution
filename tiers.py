"""tiers.py - which games may be used for what.

Drawn and recorded on 2026-10-07 in Rulebook v2's pre-registration
(docs/2-llm-history/rulebook-v2-prereg.md section 2), before any v2 model run,
and carried over unchanged into Glassbox's (docs/3-glassbox/glassbox-prereg.md
section 2). THE LISTS BELOW ARE WHAT COUNTS. draw() only shows where the two
held-out tiers came from; tests/test_tiers_registry.py checks it still
reproduces them.

  dev          building, tuning and the offline gates (Glassbox G0, G1, G2)
  seen         smoke tests (lp85, ls20); reported separately
  ab_offline   the offline held-out A/B of the templates (G3); otherwise only inside the confirm
  untouched    only ever run inside the confirm (G4)

Cross-game artifacts (Glassbox's vocabulary and templates) may be built from
`dev` and `seen` games only (BUILD_FROM).

Moved here from custom_agents/wm/tiers.py on 16 Oct 2026 (the frozen v2 copy is
legacy/rulebook/wm/tiers.py).
"""
import random

DEV = ("tu93", "tr87", "dc22", "g50t", "vc33", "ft09", "m0r0", "cd82")
SEEN = ("ls20", "lp85")
AB_OFFLINE = ("ar25", "lf52", "re86", "s5i5", "sc25", "sp80", "su15", "tn36")
UNTOUCHED = ("bp35", "cn04", "ka59", "r11l", "sb26", "sk48", "wa30")

TIERS = {"dev": DEV, "seen": SEEN, "ab_offline": AB_OFFLINE, "untouched": UNTOUCHED}
BUILD_FROM = DEV + SEEN
HELD_OUT = AB_OFFLINE + UNTOUCHED          # per-game confirm results on these are sealed
NO_ACTION7 = ("ar25", "bp35", "lf52", "sb26", "sk48", "su15")   # play with a reduced action space


def tier_of(game):
    for name, games in TIERS.items():
        if game in games:
            return name
    raise KeyError(f"{game!r} is in no tier")


def draw(seed=2026):
    """The stratified draw: both held-out tiers get 3 of the 6 games that play
    without ACTION7. Python 3.12, random.Random(2026): sample 3 of the sorted
    six, then 5 of the sorted other nine. Returns (ab_offline, untouched)."""
    r = random.Random(seed)
    six = sorted(NO_ACTION7)
    nine = sorted(g for g in HELD_OUT if g not in NO_ACTION7)
    ab = set(r.sample(six, 3)) | set(r.sample(nine, 5))
    return tuple(sorted(ab)), tuple(sorted(set(six + nine) - ab))
