"""The game tiers and the artifact registry (tiers.py, registry.py).

Copied from Rulebook v2's tests (legacy/rulebook/tests/test_wm_v2.py) when the
two modules moved to the repo root on 16 Oct 2026."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark  # noqa: E402
import registry  # noqa: E402
import tiers  # noqa: E402


# ---- tiers ----------------------------------------------------------------------------
def test_every_public_game_is_in_exactly_one_tier():
    listed = [g for games in tiers.TIERS.values() for g in games]
    assert sorted(listed) == sorted(benchmark.ALL_GAMES) and len(set(listed)) == 25
    assert tiers.tier_of("ft09") == "dev" and tiers.tier_of("wa30") == "untouched"
    assert set(tiers.BUILD_FROM) == set(tiers.DEV) | set(tiers.SEEN)


def test_the_recorded_held_out_draw_reproduces():
    ab, untouched = tiers.draw()
    assert ab == tuple(sorted(tiers.AB_OFFLINE)) and untouched == tuple(sorted(tiers.UNTOUCHED))
    assert len(set(ab) & set(tiers.NO_ACTION7)) == 3 == len(set(untouched) & set(tiers.NO_ACTION7))


def test_the_tier_lists_are_the_registered_ones():
    # Registered 2026-10-07 (v2 pre-registration section 2), carried into Glassbox unchanged.
    assert tiers.DEV == ("tu93", "tr87", "dc22", "g50t", "vc33", "ft09", "m0r0", "cd82")
    assert tiers.SEEN == ("ls20", "lp85")
    assert tiers.AB_OFFLINE == ("ar25", "lf52", "re86", "s5i5", "sc25", "sp80", "su15", "tn36")
    assert tiers.UNTOUCHED == ("bp35", "cn04", "ka59", "r11l", "sb26", "sk48", "wa30")


# ---- registry -------------------------------------------------------------------------
def test_registry_freezes_and_notices_a_change(tmp_path):
    reg = str(tmp_path / "artifacts" / "registry.json")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "a.py").write_text("x = 1\n")
    (tmp_path / "lib" / "b.py").write_text("y = 2\n")
    e = registry.add("helpers_v1", ["lib"], ["ft09", "tu93"], path=reg, root=str(tmp_path), today="2026-10-28")
    assert e["n_files"] == 2 and e["source_games"] == ["ft09", "tu93"]
    assert registry.verify(path=reg, root=str(tmp_path)) == []
    with pytest.raises(SystemExit):                              # a name is used once
        registry.add("helpers_v1", ["lib"], ["ft09"], path=reg, root=str(tmp_path))
    (tmp_path / "lib" / "a.py").write_text("x = 2\n")
    os.remove(tmp_path / "lib" / "b.py")
    bad = registry.verify(path=reg, root=str(tmp_path))
    assert sorted(b[2] for b in bad) == ["changed", "missing"]
    registry.log_event("unsealed", "confirm 1 per-game results", path=reg, today="2026-10-28")
    assert registry.load(reg)["events"][0]["what"] == "unsealed"


def test_the_registry_reads_the_repo_file():
    assert registry.PATH == os.path.join(ROOT, "artifacts", "registry.json")
    names = [a["name"] for a in registry.load()["artifacts"]]
    assert {"rulebook_v1_results", "rulebook_v2_cp0_nearmiss", "rulebook_v2_cp1_tier0a"} <= set(names)
