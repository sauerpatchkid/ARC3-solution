# Three probes before building: frozen screens, ACTION7, animation frames (10 Nov 2026)

**In one line.** The review of 10 Nov proposed giving Goose a reset watchdog,
ACTION7 (undo) and the engine's in-between frames. Three CPU probes say the
first has nothing to act on, the second is an undo that an explorer has no use
for on five of its six games, and the third matters on one game (sc25) and
would hurt on another (ft09).

**Status: probes, not tests.** No agent was changed and no GPU was used. They
size three ideas before anything is built; they do not test them.

Where the ideas come from: `future/improvement-lists-2026-11-10.md`, List 1,
items 1 and 3. Tools: `tools/frozen_moves.py`, `tools/probe_engine.py`. Tables
and commands: `results/glassbox/frozen_moves/`, `results/glassbox/engine_probe/`.

## Probe 1: is Goose ever stuck on a screen that no move changes?

**The idea tested.** Goose sends RESET only after a game over. A watchdog would
send it after about 300 moves in which nothing changed.

**What was counted.** In G4's recordings, every stretch of moves in a row that
changed no live cell, under the agent's own decoration mask rebuilt move by
move. A stretch ends at a change, at a reset and at a level-up. The watchdog's
ceiling is the part of each stretch past its first 300 moves.

| G4's runs | Moves | Moves the watchdog could have cut | Longest frozen stretch | Runs that ended frozen |
|---|---:|---:|---:|---:|
| Adopted Goose, no guide (75 runs) | 7,391,716 | **179 (0.0%)** | 379 (lp85) | 0 |
| Goose with the guide (75 runs) | 7,389,659 | **0** | 183 (lp85) | 0 |

**No run is frozen.** Outside lp85 the longest stretch in any run is 143 moves.

The two games earlier write-ups called "frozen", lf52 and sb26, are not stuck
on a dead screen. Their screen keeps changing, among very few screens: on some
seeds the scorer counts 12 distinct screens in 100,000 actions on lf52 (seeds 1
and 2) and 4 on sb26 (seed 1). Their longest frozen stretches are 65 and 64
moves. A reset goes back to a screen already seen, so it cannot open anything
there either.

**So the reset watchdog is not worth building.**

## Probe 2: what does ACTION7 do?

**The idea tested.** Six games offer ACTION7 (ar25, bp35, lf52, sb26, sk48,
su15). `run_local.py` drops it, so Goose plays them with a reduced action
space. The proposal: a sixth button.

**What was counted.** 2,000 uniformly random moves per game, straight on the
local engine, five seeds. For each use of ACTION7 after a move that changed the
screen: did it change the screen, and did it put the screen back as it was
before that move? And how many distinct screens (under the agent's mask) random
play reached with ACTION7 in the mix and without it.

| Game | ACTION7 changed the screen | It put the screen back: exactly / but for a few cells | Distinct screens, with / without ACTION7 (5 seeds) |
|---|---|---|---|
| sk48 | 99-100% | 56-67% / 26-35% | 1,049 / 1,148 · 1,006 / 1,210 · 956 / 1,201 · 1,008 / 1,203 · 995 / 1,159 |
| ar25 | 95-98% | 0% / 72-88% | 178 / 196 · 189 / 222 · 180 / 558 · 196 / 305 · 153 / 455 |
| bp35 | 100% | 0% / 59-65% | 306 / 406 · 254 / 448 · 299 / 496 · 289 / 404 · 282 / 396 |
| lf52 | 100% | 0% / 0-1% | 6 / 3 · 15 / 3 · 3 / 4 · 13 / 4 · 7 / 3 |
| su15 | 2-4% | 0% / 2-6% | 37 / 62 · 34 / 71 · 38 / 79 · 40 / 75 · 24 / 70 |
| sb26 | 0% | never | 19 / 29 · 13 / 23 · 20 / 26 · 18 / 29 · 13 / 16 |

- **On sk48, ar25 and bp35 it is an undo.** "But for a few cells" is a counter
  that ticked.
- **On sb26 it never changed the screen, and on su15 almost never.**
- **On lf52 it always changes the screen and is not an undo.** It is the only
  game where random play reached more screens with it (4 seeds of 5), and the
  numbers are tiny (3 to 15 screens).

An undo leads back to a screen already seen. Under the novelty label that is a
0, so Goose would learn not to press it. Random play shows the cost: on five of
the six games the same 2,000 moves reach fewer screens with ACTION7 than
without.

**So ACTION7 is not worth a sixth network output for the explorer.** lf52 is
the one open case: one game, on which the adopted Goose completes 1 level in 3
runs. It matters more for an agent that plans. The "LLM plays first" pilot gets
it for free, because that harness drives the engine's whole action space.

## Probe 3: do the frames before the last one hold anything?

**The idea tested.** The engine can return several frames for one move (an
animation). Goose reads only the last. The proposal: count a move as a change
when any in-between frame differs.

**What was counted.** In the same random play (seed 0), per game: how many
moves return several frames, and for the moves Goose reads as "nothing
happened" (the last frame equals the screen before the move), how many showed
something else in between.

**15 of the 25 games return several frames on some moves**: bp35 and lf52 on
every move, ft09 92%, r11l 77%, g50t 56%, sk48 53%, tu93 42%, su15 41%, sb26
34%, sc25 21%, cd82 17%, sp80 14%, tn36 3%, ls20 2%, ka59 under 1%. The other
ten never do.

On 11 of those 15 the in-between frames come only with moves that change the
screen anyway, so Goose already sees a change. The last frame hides something
on three games (and on 6 moves of ls20):

| Game | "Nothing happened" moves | ... that showed something in between | The adopted Goose's levels (G4, 3 seeds) |
|---|---:|---:|---|
| ft09 | 1,867 | 99% | 2, 2, 2 |
| sc25 | 531 | 81% | 0, 0, 0 |
| cd82 | 453 | 20% | 2, 2, 1 |

- **ft09: every dead click flashes.** The rule as proposed would label nearly
  every dead click a change, on a game where Goose does well. It would hurt.
- **sc25 is the open case.** Goose completes no level there in any arm, and
  four of five moves that end where they started show something in between.
  What those frames show has not been looked at.
- tn36, which the review named, returns several frames on 3% of random moves.

**So the proposal as worded should not be built.** One look at sc25's
in-between frames comes first.

## What this means for the week

Item 4 of this week's list was "a reset watchdog and ACTION7". Both halves have
little to act on. The time-to-level re-score
(`docs/3-glassbox/time-to-level.md`) points somewhere else: the guide's
LLM-free carry makes levels after the first come sooner. That is the review's
List 1, item 2 (carry colour no-ops and goal hints inside Goose), which was not
on the list. Whether it replaces item 4 is Matt's call.

## Limits

- **Random play is not Goose.** It shows what the engine can return, not what
  an agent would gain.
- **Probe 1 sees what the agent's mask sees.** A decoration the mask misses
  makes every move look like a change, so a dead screen behind it would not be
  counted. The watchdog would not see it either.
- **Probe 3 is one seed**, and counts frames, not what they mean.
