"""tools/levels_by_budget.py: levels by action budget and the level-by-level races, on made-up runs."""
import json

import levels_by_budget as L


def make_runs(tmp_path, arm, runs):
    """A manifest of one arm; `runs` is {(game, seed): ([level-up actions], actions taken)}."""
    rows = []
    for (game, seed), (ups, taken) in runs.items():
        d = tmp_path / arm / f"{game}_s{seed}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({
            "levelup_events": [[a, i + 1] for i, a in enumerate(ups)], "actions_taken": taken,
            "n_actions": taken, "levels_completed": len(ups)}))
        rows.append(f"{d}\t{game}\t{seed}\t{arm}")
    path = tmp_path / f"{arm}.tsv"
    path.write_text("\n".join(rows) + "\n")
    return f"{path}:{arm}"


BASE = {("g1", "0"): ([100, 500], 1000),        # two levels
        ("g2", "0"): ([100], 1000),             # one level
        ("g3", "0"): ([], 1000)}                # none
NEW = {("g1", "0"): ([80, 300, 900], 1000),     # faster on levels 1 and 2, and a third level
       ("g2", "0"): ([], 1000),                 # never finished level 1
       ("g3", "0"): ([], 1000)}


def load_both(tmp_path):
    return L.load(make_runs(tmp_path, "base", BASE)), L.load(make_runs(tmp_path, "new", NEW))


def test_levels_by_budget_counts_what_was_done_by_then(tmp_path):
    base, new = load_both(tmp_path)
    at = lambda t: L.paired(base, new, lambda run: L.level_at(run[0], t))
    assert (at(90)["base"], at(90)["new"]) == (0, 1)
    assert (at(600)["base"], at(600)["new"]) == (3, 2)      # base: g1 2 + g2 1; new: g1 2
    r = at(1000)
    assert (r["base"], r["new"]) == (3, 3) and (r["wins"], r["ties"], r["losses"]) == (1, 1, 1)


def test_races_count_a_level_one_run_never_finished(tmp_path):
    base, new = load_both(tmp_path)
    races = {(g, level): (winner, ratio) for g, _, level, winner, ratio in L.level_races(base, new)}
    assert races[("g1", 1)] == ("new", 0.8)                 # 80 against 100 actions
    assert races[("g1", 2)] == ("new", 0.55)                # 220 against 400 actions on the level
    # level 3: new took 600 actions; base had only 500 left on it and did not finish: undecided
    assert races[("g1", 3)] == ("undecided", None)
    # g2 level 1: base took 100; new spent all 1,000 without finishing: base was faster
    assert races[("g2", 1)] == ("base", None)
    assert not any(g == "g3" for g, _ in races)             # neither finished anything: no race


def test_a_run_that_stopped_early_keeps_its_levels_and_is_named(tmp_path):
    base = L.load(make_runs(tmp_path, "base", {("g1", "0"): ([100, 200], 300)}))    # won at 300 actions
    new = L.load(make_runs(tmp_path, "new", {("g1", "0"): ([150], 1000)}))
    text = L.report(base, new, budgets=[200, 1000])
    assert "base g1 seed 0 (300 actions, 2 levels)" in text
    assert "| 1,000 actions | 2 | 1 | -1 |" in text


def test_only_common_pairs_and_the_games_filter(tmp_path):
    base = L.load(make_runs(tmp_path, "base", BASE), games={"g1", "g2"})
    new = L.load(make_runs(tmp_path, "new", {k: v for k, v in NEW.items() if k[0] != "g2"}))
    assert set(base) == {("g1", "0"), ("g2", "0")}
    assert L.report(base, new).startswith("Pairs compared: 1 over 1 games.")
