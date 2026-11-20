# Knowledge Planner, Phase 2: where it works and where it does not (19 Nov 2026)

**In one line.** Of the six dev and seen games that offer clicks, the planner
reaches two, ft09 and lp85. There it clears every level it has a rule for:
all five later levels of ft09 from 20 of 20 recordings, and levels 2 to 4 of
lp85 from 15 of 15, where it then stops on level 5. On cd82, dc22, m0r0 and
vc33 it never starts, because their screens do not stay on a level's slots and
level 1 teaches it nothing. No setting was adopted, so the planner that goes
into the test is the groupmate's E20, unmodified.

**Status: exploratory.** CPU only, no new run of any game, tuned tier (these
are the games the plan allows building on). It is the plan's "likeliest
outcome", and it says what kind of game the method reaches: puzzles whose
objects stay put and only change, with the solved board shown before the next
level.

The plan: `kp-plan.md`, Phase 2. The tool: `tools/kp_offline.py`. Tables:
`results/kp/phase2/`; the setting trial: `results/kp/phase2_mask/`.

**How it was measured.** Every unguided recording made since 6 Oct, on the six
click games: 116 recordings (upgrade screen 3's nine arms, the adopted Goose in
G4, dev sweep 2 and the carry test). Each is replayed on the engine, with the
real bridge (`custom_agents/kp_bridge.py`) getting the calls Goose's hook
makes and the recording standing in for Goose. Two ways:

- **takeover:** the recording plays level 1, then the planner plays alone to
  action 5,000. The episode stops when the planner hands the level back.
- **induct:** the planner watches the whole recording and never takes a move.
  This shows what it could learn at every level Goose itself cleared.

## 1. The funnel: the levels after the first

From the takeover. Each step counts a level only if it passed every step
before it.

| Game | Levels entered | Had a rule | Model settled | Plan found | Cleared by planning | Not cleared | Where it stopped |
|---|---|---|---|---|---|---|---|
| **ft09** | 100 | 100 | 100 | 100 | **100** | 0 | |
| **lp85** | 60 | 60 | 60 | 60 | **45** | 15 | 41 wrong predictions, on level 5 (15) |
| cd82 | 17 | 0 | | | 0 | 17 | no clickable kind learned (17) |
| dc22 | 9 | 0 | | | 0 | 9 | no clickable kind learned (9) |
| m0r0 | 20 | 0 | | | 0 | 20 | no clickable kind learned (20) |
| vc33 | 20 | 0 | | | 0 | 20 | no clickable kind learned (20) |

- **Had a rule:** the planner had a clickable kind, a click effect and a
  confirmed rule, and engaged. **Model settled:** probing and verifying were
  over and it chose a rule to plan to. **Plan found:** a solver returned a
  plan. **Cleared by planning:** the level ended with the planner in charge.
- The plan's step "slots held" is left out of this table: on a level the
  planner never engaged on, the takeover has one screen, so the step says
  nothing. Section 3 has that share over Goose's whole levels.
- Offline nothing is "cleared by Goose after a fallback": the episode stops
  when the planner gives a level up. In a real run Goose plays on.

## 2. The planner alone, beside the recording's Goose

Levels completed by action 5,000, written as `levels: number of prefixes`.

| Game | Prefixes | The planner, from Goose's level 1 | The recording's Goose |
|---|---|---|---|
| ft09 | 20 | 6: 20 | 1: 16, 2: 4 |
| lp85 | 15 | 4: 15 | 1: 7, 2: 6, 3: 1, 5: 1 |
| cd82 | 17 | 1: 17 | 1: 16, 2: 1 |
| dc22 | 9 | 1: 9 | 1: 9 |
| m0r0 | 20 | 1: 20 | 1: 20 |
| vc33 | 20 | 1: 20 | 2: 18, 3: 2 |

Fifteen recordings have no level 1 inside 5,000 actions and are no prefix: 11
on dc22, 3 on cd82, 1 on lp85.

On vc33 Goose alone ends on 2 or 3 levels and the planner adds nothing. That
costs nothing in a real run: where the planner does not start, the agent is
Goose (the passive CPU check of Phase 1 shows it move for move).

## 3. What the planner could learn at each level clear

From the induct pass: 116 recordings watched, 140 level clears, no error and
no replay mismatch.

| Game | Level | Clears seen | Screens that fit the slots | Objects on the first screen | Click effect seen | Solved board fits | Clears with a consistent rule | Rule carried in: confirmed again |
|---|---|---|---|---|---|---|---|---|
| ft09 | 1 | 20 | 100% | 40 | recolour, cycle of 2 | 20 of 20 | 20 of 20 | |
| ft09 | 2 | 4 | 100% | 16 | recolour, cycle of 2 | 4 of 4 | 4 of 4 | 4 of 4 |
| lp85 | 1 | 15 | 100% | 34 | move | 15 of 15 | 15 of 15 | |
| lp85 | 2 | 8 | 100% | 64 | move | 8 of 8 | 8 of 8 | 8 of 8 |
| lp85 | 3 to 5 | 4 | 100% | 43 to 66 | move | 4 of 4 | 4 of 4 | 4 of 4 |
| cd82 | 1 | 17 | 8% | 6 | none | 0 of 17 | 0 of 17 | |
| cd82 | 2 | 1 | 6% | 7 | none | 0 of 1 | 0 of 1 | |
| dc22 | 1 | 9 | 30% | 20 | none | 2 of 9 | 0 of 9 | |
| m0r0 | 1 | 20 | 2% | 2 | none | 0 of 20 | 0 of 20 | |
| vc33 | 1 | 20 | 65% | 2 | none | 0 of 20 | 0 of 20 | |
| vc33 | 2 | 20 | 43% | 3 | none | 0 of 20 | 0 of 20 | |
| vc33 | 3 | 2 | 11% | 1 | mixed | 0 of 2 | 0 of 2 | |

On ft09 and lp85 every step of the learning holds at every clear seen: the
screens fit, the solved board fits, at least one rule is left standing, and the
rule carried in from the level before is confirmed again (16 of 16). The top
rule is "every multi-colour pattern describes its neighbourhood" on ft09 and
"for every mark, some nearest dynamic object has the same colour" on lp85.

## 4. Game by game: where it stops and why

- **ft09.** Nothing stops it. Tiles sit on a lattice and only change colour.
- **lp85.** It plans levels 2, 3 and 4 and gives level 5 up after 41 wrong
  predictions, the reference's cap of 40. Level 5 has more buttons than its
  model settles in that many tries. In a real run Goose takes level 5 back.
- **vc33.** The reference sees 2 or 3 objects on a level's first screen: most
  of the screen is large one-colour areas, which it counts as background. A
  click resizes or moves what is left, so 35% of level 1's screens and 57% of
  level 2's are off the slots, no click is seen to do anything, and the solved
  board never fits (0 of 42).
- **m0r0.** Two objects on the first screen and 2% of screens on the slots.
  It is mostly a keyboard game (clicks were 8% and 10% of level 1's moves in
  two recordings), and its pieces move.
- **cd82.** Level 1 is nine or ten moves, all key presses in the two
  recordings looked at, and the pieces move: 8% of screens fit.
- **dc22.** Twenty objects, 30% of screens on the slots. Pieces move and
  appear. The solved board fits at 2 of 9 clears, but with no click seen to do
  anything, no rule is left standing.

So the four have one cause: **things move or change size.** The planner's
state is "which signature sits on which fixed slot", and it reads a screen
only when every object is on a slot.

## 5. The settings: one tried, none adopted

The adoption rule, fixed in the plan before this was run: a setting is
switched on only if, on this takeover table, it clears strictly more levels by
planning and loses none.

| Setting | The plan's trigger | Shown here? | Trial | Adopted |
|---|---|---|---|---|
| `mask = "agent"`: paint Goose's decoration mask over a screen before it is parsed | a game losing screens to a bar or a counter | In part. With the mask painted over, dc22's level-1 screens fit 34% and 56% of the time, against 23% and 36% (a side check on two recordings). The other three games did not change | **Tried on all 101 prefixes, with the mask as Goose's hook passes it: 172 levels against 246 without it.** ft09 fell from 20 of 20 games won to 4 of 20. lp85 reached level 5 from 6 prefixes (none without it) and lost levels on 2. dc22 and the other three: no change | **No.** It loses levels |
| `win_frame = "last_valid"`: use an earlier frame of the level-completing move as the solved board | a clear with no usable solved board although an earlier frame fits | No. Phase 0: on dc22, m0r0 and vc33 the move returns two frames only, and on cd82 none of the 15 earlier frames fits | Not built | |
| `try_unconfirmed = N`: try rules that are not confirmed yet | the planner idle on a level whose slots held, for want of a confirmed rule | No. The slots held only on ft09 and lp85, and there a rule was confirmed at every clear | Not built | |

**Why the mask hurts on ft09.** Goose's mask grows while a level is played
(its detector looks again every 250 moves, and bars are added as they are
seen). A level's slots are fixed from its first screen. So later screens are
parsed under a different mask than the slots were, and stop fitting: 12 of the
20 prefixes end with no rule confirmed at level 1's clear, and 4 with a screen
that left the slots.

`mask` stays in `experiments/kp/kp.toml` as `"none"`, with its test, in the
way `deadclick` stays among Goose's switches. The other two settings are not
in the file: every setting there is read by code.

## 6. The bridge gives the reference's own results

Phase 0 ran the takeover on ft09 and lp85 with the reference in its own
harness. This phase ran it through the bridge. The 36 rows agree in every
column: the action of level 1, the levels reached, the actions for each
planned level, and how it stopped. Through the bridge the 36 episodes took 555
seconds against 2,035, because a screen is parsed once.

## What it means for the test

- **The gain can come from ft09 and lp85 only, among these ten games.** The
  15 held-out games are not touched until the test. A game there counts for
  the planner only if its pieces stay on their slots.
- **lp85 is still the uncertain one.** The planner stops at level 5, and
  Goose alone sometimes gets further (one recording reached 5 levels).
- **Nothing was tuned.** The test's planner is E20 as received.

## Run it again

```bash
uv run python tools/kp_offline.py takeover          # about 6 minutes with 4 at a time; writes the funnel too
uv run python tools/kp_offline.py induct            # about 20 minutes with 4 at a time, beside other jobs
uv run python tools/kp_offline.py takeover --config <a .toml with [see] mask = "agent"> --out results/kp/phase2_mask
```
