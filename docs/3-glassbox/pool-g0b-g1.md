# A pooled book: G0b's and G1's rules together — exploratory (18 Oct 2026)

*Exploratory: no pre-registered section judges it, and it changes no verdict.
G1 stays a FAIL (3 of 8; `g1-offline-rules.md`). This is the follow-up Matt
chose on 18 Oct ("(a) then (b)"): (a) one book from both runs' rules, no LLM;
(b) S4's role tie-break, tested on the same books.*

## In one line

**A book drawn from both runs' candidates keeps the best of each: on the
training level it is never below the better of G0b and G1 and is above both on
3 games (dc22 40.4%, g50t 18.1%, cd82 6.6%); on the next level (T1) it carries
exactly as well as the better source on every game.** It costs no LLM call.
It is the natural book for the offline counterfactual and for G2
(pre-registration section 5, OPEN (a)).

## What was run

- **S2 POOL** (`custom_agents/gb/s2_pool.py`, new): for each dev game, G0b's
  candidates (CP1's rules, as imported) and G1's, plus the narrowed rules
  G1's guard fitter trusted (cd82 and ft09, one each), renumbered into one
  `candidates.jsonl`. The two runs' evidence is the same file, byte for byte
  (S0 on, CP1's levels and splits), which the pool checks before it starts.
  No candidate appears twice. Groups: G0b's order, then any group only G1
  called.
- Then the stages unchanged: S1, **S3's grading** (each candidate re-graded
  with its own binding; the book chosen by CP1's procedure: per group, in
  order, lint-clean first, most gain, at most 4, a later rule must add cells)
  and **S4** (CP1's re-binding on the transfer level's 300-move fit split; T1
  on the score split). Nothing was asked again and nothing re-fitted.
- `make gb-pool` (POOL_RUN=`pool_g0b_g1`); `results/glassbox/pool_g0b_g1/`,
  `pool_report.md` there. 18 Oct, code as committed with this write-up. S4 ran
  on a quiet machine: no binding search was cut short.

## Per game

Training level: the composed book's cell coverage (case coverage in brackets);
**bold** = above both sources. Transfer level: T1 per move, with "nothing" and
"memory"; wrong = share of the score split's changing cases with a wrong
claimed cell; "beats both" = T1 above "nothing" and "memory" with both 95%
intervals above zero (CP1's criterion).

| Game | G0b | G1 | Pool | T1 pool (nothing, memory) | T1 G0b | T1 G1 | Wrong | Beats both | Book: rules from |
|---|---|---|---|---|---|---|---|---|---|
| tu93 | 0.0% (0.0%) | 59.0% (59.0%) | 59.0% (59.0%) | 40.7% (3.8%, 87.1%) | 3.8% | 40.7% | 0% | no | G1 ×3 |
| tr87 | 0.0% (0.0%) | 35.5% (25.1%) | 35.5% (25.1%) | 19.7% (0.0%, 47.6%) | 0.0% | 19.7% | 0% | no | G1 ×1 |
| dc22 | 11.8% (14.7%) | 29.3% (6.0%) | **40.4%** (20.8%) | 25.5% (15.4%, 23.4%) | 25.5% | 15.4% | 0% | **yes** | G0b ×3, G1 ×2 |
| g50t | 17.7% (0.0%) | 9.0% (0.0%) | **18.1%** (0.0%) | 9.1% (9.1%, 15.0%) | 9.1% | 9.1% | 0% | no | G0b ×2, G1 ×1 |
| vc33 | 10.6% (0.0%) | 6.1% (0.0%) | 10.6% (0.0%) | 75.1% (75.1%, 75.1%) | 75.1% | 75.1% | 0% | no | G0b ×1 |
| ft09 | 97.7% (56.0%) | 59.2% (56.0%) | 97.7% (56.0%) | 18.6% (3.0%, 15.2%) | 18.6% | 3.0% | 0% | **yes** | G0b ×3 |
| m0r0 | 53.5% (53.0%) | 53.5% (53.0%) | 53.5% (53.0%) | 51.7% (6.8%, 53.9%) | 51.7% | 6.8% | 4.26% | no | G0b ×2 |
| cd82 | 5.2% (8.3%) | 4.2% (4.0%) | **6.6%** (8.3%) | 14.9% (13.1%, 17.3%) | 14.9% | 13.7% | 0.15% | no | G0b ×3, G1 ×5 |

## What it shows

**1. The two runs' rules are complementary, not competing.** Where one run had
nothing (tu93, tr87: G0b; ft09's red-click group: G1), the pool takes the
other's. Where both had rules for different parts of a game, it takes both:
dc22's book holds 3 of CP1's rules and 2 of G1's and explains 40.4% of the
changed cells, more than either alone (11.8%, 29.3%); cd82's holds 3 and 5.

**2. Pooling never cost coverage here.** Because CP1's procedure orders rules by
gain within a group and admits a later rule only if it adds cells, the pooled
book is at least as good as the better source on all 8 games. That is a
property of this procedure on these runs, not a guarantee: a rule that claims
cells another rule claims differently would make them UNKNOWN.

**3. Transfer is the better source's, game by game.** T1 equals G0b's on the
six games where G0b's rules lead the book (dc22, g50t, vc33, ft09, m0r0, cd82)
and G1's on tu93 and tr87. dc22's extra training-level
coverage does not show at T1 (25.5%, as G0b): G1's two dc22 rules add cells on
level 1 but carry to level 2 no better than before. Safety is G0b's: m0r0's
4.26% wrong is CP1's rules' (registered in G0b), cd82's 0.15% is G1's.

**4. Read against G1's primary, the pool would be 5 of 8 above G0b** (tu93,
tr87, dc22, g50t, cd82). It is not G1, it was not registered, and its idea
(keep earlier rules) came from looking at G1's result, so it is reported only
as what a book across runs looks like.

## (b) S4's role tie-break

**The problem.** G1's m0r0 rules carried to nothing: on level 2's fit split,
113 bindings of its ACTION1 rule were exact and all applied to the same 22
cases, and the search's first (the one used) moved `agent` from the mover
(colour 10) to a colour that is static on level 2 (6). Its T1 was "nothing"
(6.8%).

**The change** (S4 v1, `[carry] roles_in_bind`, off by default): among the exact
bindings that apply to the most fit cases, prefer the one whose colour slots
keep their S1 role (mover, static, click changer, click no-op, counter) from
the training level to the next. Runs `pool_g1_roles` (G1's candidates alone)
and `pool_g0b_g1_roles` (the pool), S1 → S3 → S4 with the knob on, on the same
evidence; everything else as above.

| Book | Game | T1, roles off | T1, roles on | Wrong (on) |
|---|---|---|---|---|
| G1 alone | m0r0 | 6.8% | **26.5%** | 0.8% |
| G1 alone | the other 7 | as G1 | unchanged | as G1 |
| pool | all 8 | as above | unchanged | as above |

On m0r0 the role binding keeps `agent` on the mover and moves `step` 5 → 4,
as CP1's own rule did (G0b: 51.7%). The second m0r0 rule (ACTION2) has no
exact binding among those applying to the most cases that keeps `agent` on the
mover, so it keeps the search's choice, and the book reaches 26.5%, not 51.7%. On the pool nothing changes because its m0r0 rules
are CP1's, whose search choice already keeps the roles.

**Tuned on dev games, labelled as such.** The first version let roles override
the number of cases a binding applies to; on the pool it cost ft09 3.9 points
of T1 (18.6% → 14.7%) by choosing a colour that applied to 2 fit cases over
one that applied to 6, because S1's role sets for the two colours differ only
by "counter". The version kept never prefers a binding that applies to fewer
cases (`tests/test_gb_carry.py` pins that case). Recommended setting for G2:
on (it can only reorder exact bindings among equals); offline it changes
nothing for the pooled book. S5's online re-binding of no-op rules does not
use it yet.

## What it means for M2

- **The book for the counterfactual and G2:** this pooled book is the
  recommended choice (section 5, OPEN (a)); it holds every trusted rule either
  run found, with G0b's transfer on the games G0b carried and G1's on tu93 and
  tr87.
- **What S5 v1 uses from it:** only its no-op rules (per group, the known
  no-op with the most moves: dc22 2, ft09 2, m0r0 1, cd82 2, g50t 1, vc33 1;
  none on tu93 and tr87). Its changing rules wait for probes or IMAGINE.
- **IMAGINE's precondition** (plan §7.3: on some dev game, rules trusted on two
  levels predict the whole live board on ≥ 20% of a level's cases): on tu93 the
  carried rules get the whole screen right on 40.7% of the transfer level's
  moves against 3.8% for "nothing", and on m0r0 51.7% against 6.8%, so it
  holds on at least tu93 and m0r0 by this reading (the T1 share of moves
  whose whole screen is right, per move, minus nothing's floor).
