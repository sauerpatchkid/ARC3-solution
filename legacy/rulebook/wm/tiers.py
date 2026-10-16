"""tiers.py - which games may be used for what (docs/plans/rulebook-v2.md section 2.4).

Recorded 2026-10-07 in docs/plans/rulebook-v2-prereg.md, before any v2 model run.
THE LISTS BELOW ARE WHAT COUNTS. draw() only shows where the two held-out tiers
came from; the test in tests/test_wm_v2.py checks it still reproduces them.

  dev          building, tuning and the offline gates of CP1 and CP2
  seen         used by Stage A or the smoke test; reported separately
  ab_offline   the offline held-out A/Bs of CP3 and CP5; otherwise only inside a confirm
  untouched    only ever run inside a confirm

Cross-game artifacts (helpers, schemas, the playbook, a prior, a fine-tuned
writer) may be built from `dev` and `seen` games only (BUILD_FROM).
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
