"""tools/gb_funnel.py: the funnel's numbers against G1's registered ones (where the run is on this
machine), and the online row's arithmetic."""
import collections
import os
import statistics as st

import pytest

import gb_funnel as F
from gb import artifacts as A

G1 = os.path.join(A.results_dir(), "g1")


@pytest.mark.skipif(not os.path.isdir(os.path.join(G1, "s3_check")), reason="results/glassbox/g1 is not on this machine")
def test_g1_rows_give_the_numbers_g1_published():
    """docs/3-glassbox/g1-offline-rules.md and results/glassbox/g1/g1_verdict.md."""
    rows = {g: F.game_row(G1, g) for g in F.games_of(G1)}
    assert len(rows) == 8
    assert round(st.mean(r["cell"] for r in rows.values()), 3) == 0.320          # mean cell coverage 32.0%
    assert sum(r["trusted"] for r in rows.values()) == 32
    assert sum(r["answers"] for r in rows.values()) == 385
    assert sum(r["ran_out"] for r in rows.values()) == 134                       # 35% of answers out of room
    tu93 = rows["tu93"]
    assert (tu93["cell"], tu93["t1"], tu93["nothing"], tu93["memory"]) == (0.5902, 0.4073, 0.0383, 0.8712)
    assert tu93["whole"] <= tu93["part"] <= tu93["admitted_cases"] <= tu93["called"] <= tu93["changing"]


def test_online_row_rule_survival_is_the_share_never_demoted():
    c = collections.Counter({"runs": 3, "moves": 1000, "changed": 250, "colour no-op carried": 7,
                             "colour no-op dropped": 1, "rule re-bound": 10, "rule demoted": 2,
                             "goal hints": 4, "goal hint dropped": 3})
    assert F._online_row("vc33", c) == "| vc33 | 3 | 25% | 7 | 1 | 10 | 2 | 80% | 4 | 3 |"
    assert F._online_row("tu93", collections.Counter(runs=3)) == "| tu93 | 3 | - | 0 | 0 | 0 | 0 | - | 0 | 0 |"
