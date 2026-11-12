# ARC3-solution: where it stands and what to try next

10 Nov 2026 · prepared by Claude for Matt · editable copy: https://claude.ai/artifact/Chx4zaap9A4kua9EhcythY

## Bottom line

Glassbox's LLM adds little because of the job it was given, more than how well it does that job. It writes transition rules offline. Only its "this action does nothing" rules reach Goose, on 6 games, and an LLM-free colour rule already covers almost all of those moves. The published LLM gains come from a different job: the model plays the game itself, guesses the goal, and is scored on how few actions it needs.

- **The explorer result is solid.** 54 to 79 to 112 levels is about 20% of the 549 available. Keep it as the report's lead.
- **The guided agent's gain is small and LLM-free.** All three guided-vs-unguided sweeps favour the guide (+1, +5, +9 levels), each inside its own noise band. Outside the 8 dev games the guide used no LLM output.
- **The outside jump came mostly from stronger models in a "model plays" harness.** One open harness scored 1.21% on the hidden Kaggle games with a 27B Qwen and 27.89% three months later with Qwen3.8-Flash-Next. That model needs about 74 GB of GPU memory, but its hosted version costs $0.15 / $0.47 per million tokens.
- **Our headline metric hides what LLMs are good at.** Levels at 100,000 actions rewards brute-force coverage. Published scores reward finishing in few actions.

**Recommended first step:** before building anything, re-score the 150 G4 runs on time-to-level and count frozen runs in them. Both need no GPU. Then spend the week on one Goose fix (ACTION7 plus a reset watchdog) and one LLM test (a stronger rule writer through the API, then either a goal test or the open "model plays" harness as a reference arm). The last section gives the order.

## What the repo shows

The one large, well-supported gain is the LLM-free explorer work: 54 to 79 to 112 levels. Every LLM stage since has failed its gate or passed inside the noise. Read from the `glassbox` branch at `c9f59df` (8 Nov); `main` is still the July code. Levels are summed over 25 public games x 3 seeds x 100,000 actions unless a row says otherwise.

| Stage (date) | What it does | Result | Verdict |
| --- | --- | --- | --- |
| Goose baseline (semester 1) | CNN trained online on "did this action change the frame?"; samples actions from that prediction | 54 levels; 11 of 25 games with a level | reference |
| Novelty label (22 Sep) | Label becomes "reached a screen not seen on this level", decorations masked | 79 levels; paired 23 better / 51 same / 1 worse | adopted |
| `mb_gated_att` = A0 (28 Sep) | Adds a return map, masked progress bars, a bandit that gates walk-backs, and half credit per attempt | 112 levels; 31 / 36 / 8; 22 of 25 games | adopted |
| LLM judge and heuristic writer (Sep, 2-9B models) | LLM ranks which move is progress; LLM writes scoring code | Judge at chance (0.50); heuristics never beat a one-line rule | NO-GO |
| Coach (3-4 Oct) | On a stall, the 27B reads a text summary and ranks what to try | 21 hits vs 21.2 expected by chance | NO-GO |
| Rulebook v1 (4-6 Oct) | 27B writes exact transition rules, replay-checked | Stage A 1 of 3 games; Tier 0a 2 of 8; ft09 rule 58.8% wrong on the next level | FAIL |
| Rulebook v2 (7-15 Oct) | Rules take level facts as parameters, re-bound per level without the LLM | CP1 2 of 8; 0.0% wrong after re-binding | FAIL (transfer fixed, too few rules) |
| Glassbox offline: G1 (17 Oct), G3 (3 Nov) | Adds object facts, a primitive library, one mechanism per rule, templates across games | G1 3 of 8 above the earlier book; mean cell coverage 32.0% on dev games, 13.8% on held-out games; templates adopted 0 of 792 tries | FAIL, FAIL |
| Glassbox online: G2 (19 Oct) | Goose guided by S5 v1 on 8 dev games x 2 seeds | 34 vs 37 levels; LLM-free arm 35 | FAIL |
| Glassbox online: dev sweep 2 (5 Nov) | S5 v2 vs a same-day A0, 8 dev games x 4 seeds | 72 vs 67; 6 / 24 / 2 | PASS (about 1.6 SD) |
| Glassbox online: G4 (8 Nov) | S5 v2 vs a same-day A0, all 25 games x 3 seeds | 110 vs 101; 12 / 57 / 6; same-day A0 was 11 below its own 27 Sep total of 112 | ADOPT (about 1.1 SD) |

What the guided agent actually uses online: colour no-ops carried across levels and goal hints (both LLM-free), plus the book's no-op rules on 6 dev games. The LLM's rules about what actions *do* are never used to choose a move. Outside the 8 dev games the guide ran with no LLM output at all, and that is where 5 of the 9 extra levels came from.

## How our setup differs from published work

Published LLM results differ from ours in three things at once: what the LLM does, how strong it is, and what is scored. Scores below are RHAE, the official action-efficiency score, unless a row says otherwise.

| System (date) | What the LLM does | Model | Score and setting |
| --- | --- | --- | --- |
| [StochasticGoose](https://arxiv.org/html/2603.24621v1), preview competition (2025) | no LLM | CNN explorer | 12.58%, first place, on the preview games |
| [Frontier models, no harness](https://arxiv.org/html/2603.24621v1) (Mar 2026) | plays directly | GPT-5.4, Opus 4.6, Gemini 3.1 Pro | 0.25% to 0.37%, semi-private set |
| [Duck harness](https://tufalabs.ai/research/duck-harness/), Milestone 1 (30 Jun) | plays through a Python REPL | Qwen 3.6 27B, local | 1.21% on Kaggle's hidden games; 1.60 +/- 0.45 on the 25 public games |
| [Franzen](https://github.com/da-fr/arc-agi-3-solution), Milestone 2 (30 Sep) | same harness, tuned | Qwen3.8-Flash-Next on one 96 GB GPU | 27.89% hidden. On the 25 public games about 45 averaged over 4 passes and about 60 taking the best of 4 ([replication 1](https://github.com/hammadfarooq-ai/arc-prize-2026-arc-agi-3), [replication 2](https://github.com/veeda241/ARC-Prize-2026---ARC-AGI-3)) |
| [Kaggle public leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard) at close (2 Nov) | top teams' methods not yet published | open weights, offline | top score 75.56; final results due 4 Dec |
| Non-LLM explorers on the same Kaggle board ([graph explorer](https://github.com/hammadfarooq-ai/arc-prize-2026-arc-agi-3), [algorithmic line](https://github.com/Sahasawatt/arc-agi-3-agent)) | no LLM | graph search | 0.1% to 0.3% |
| [Rodionov](https://arxiv.org/abs/2605.05138) and his [component study](https://arxiv.org/abs/2607.15439) (May to Jul) | writes a Python simulator, verifies it by replay, plans through it | GPT-5.5, then GPT-5.6 Sol | 58.12% with 15 games solved; about 99% with GPT-5.6 Sol (public games) |
| [Tycho](https://arxiv.org/abs/2607.28287) (Jul) | builds a model only when the acting agent asks for one | Opus 4.8 | 88.49; 79.07 with no world model; 83.07 with automatic repair (public games) |
| [Official leaderboard](https://arcprize.org/leaderboard) (read 10 Nov) | plays; the Standard harness lets it keep notes | Claude Fable 5.2, GPT-6 Astra, Gemini 4 Argon | 81.2%, 62.7%, 49.8% semi-private. Best open-weights entry 0.9% |

The same comparison, attribute by attribute:

|  | Published LLM harnesses | This project (Glassbox) |
| --- | --- | --- |
| The LLM's job | Plays: looks at the board, runs code on it, tests a guess, picks the actions | Writes per-action transition rules offline from Goose's recordings; never picks an action |
| What reaches the agent | Every move | "This does nothing" on 6 dev games; nothing on the other 19 |
| Who finds the goal | The model guesses it and tests it | No LLM stage. One LLM-free rule: the two colours that shrank most before the last win |
| Model | Flash-Next (125B, about 74 GB at 4-bit) or a frontier model | Qwen3.8-27B at 4-bit on a 32 GB card |
| Score | (human actions / agent actions) squared per level, cut off at 5x human | Levels completed in 100,000 actions per game |
| Actions per level | Tens to hundreds | Thousands to tens of thousands |
| What is checked | Nothing (Duck), or whole-game replay (Rodionov, OPINE-World) | Every rule, cell by cell, on every recorded case |
| Inputs | 10x images, difference images, animation frames, UNDO | Last frame of each animation only; ACTION7 filtered out on 6 games |

Four readings of these tables:

- **Model strength is the largest single factor.** The Duck harness went from 1.21% to 27.89% when the 27B was replaced by Flash-Next and the harness was tuned. Rodionov found every variant improves with a stronger model, often by more than the variants differ. The Standard-harness leaderboard went from under 1% in March to 81.2% in November.
- **For a 27B-class model, imposed structure does not help.** [ARC Prize's Milestone 1 notes](https://arcprize.org/blog/arc-prize-2026-milestone-1) report that hand-built tools hurt the Duck's 27B. Franzen's best change in that area was switching the structured world model off. [Another team](https://github.com/pcurran100/arc-agi-3) built "a 27B writes a verified world model" and scored the same as its no-LLM explorer, because 34 of 45 answers were truncated or degenerate. That is our overrun problem (35% to 46% of answers), found independently.
- **Exact transition match is not what limits even strong models.** Tycho's automatic repair made models that match recorded transitions better and scored lower (83.07 vs 88.49). With no world model at all it scored 79.07. Its authors conclude that matching transitions does not show the objective was found.
- **The two settings measure different things.** Under RHAE an explorer scores near zero: 0.1% to 0.3% on Kaggle, and about 0 for every arm of ours (CLAUDE.md). Under levels at 100,000 actions an explorer is strong: A0 completes about 37 of 183 levels per seed, where a no-LLM graph agent scored under RHAE completes 9 to 10 ([same team](https://github.com/pcurran100/arc-agi-3)). So "LLM agents beat explorers a hundredfold" and "our LLM adds nothing to the explorer" are both true.

How context is kept also moves scores a lot: [GPT-6 Astra](https://arcprize.org/blog/astra) scores 62.7% with the Standard harness and 99.9% with its provider's own context handling.

## Why Glassbox is not paying off

Five causes, in order of how much each one costs. Each number is from the repo file named beside it.

1. **The LLM's output barely touches the agent.** Online, S5 uses only the book's no-op rules to prune moves (`gb/s5_act.py`). In the offline counterfactual they added at most 3.6 points of covered moves over the LLM-free colour rule (`cf-counterfactual.md`). The guided arm and its LLM-free twin tied in G2, 34 vs 35 levels (`g2-online.md`). In G4 the book had rules for 6 games and none for the other 19 (`g4-confirm.md`).
2. **No stage finds the goal.** Rules say what an action does. Nothing says what to aim for. The imagined-move search was dropped on 2 Nov because no game met its precondition, and contact goals went with it (`carry-later-levels.md`). The one goal signal is the LLM-free two-colour hint, and G4's gain sits on the games where it acted: s5i5 +3, r11l +2.
3. **The 27B is too weak for exact rule writing at this budget.** Mean cell coverage is 32.0% on the dev games and 13.8% on held-out games. Between 35% and 46% of answers ran out of room. Each game takes about one GPU-hour (`g1-offline-rules.md`, `g3-templates.md`).
4. **Rules do not hold where Goose stops.** On the level where runs end, the carried book is right wherever it applies only on ft09 and tr87. tu93 level 5 and m0r0 level 3 bring mechanics the book never saw, and vc33's rule never re-binds (`carry-later-levels.md`). Across games nothing carried: 0 of 792 template tries (`g3-templates.md`).
5. **The test cannot see small effects.** Two unguided 75-run totals differ by about 8 levels (1 SD), and the confirm rule passes 36% of the time with no effect (`g4-confirm.md`). The guide acts in the first 2,000 moves of a new level, so it should shift time-to-level more than the level count at 100,000 actions. Time-to-level was never scored.

Three things worked and are worth keeping in the report as results. Replay checking plus re-binding gave 0.0% wrong predictions on the next level in CP1. The guide (carried colour no-ops and goal hints, LLM-free on 19 of 25 games) is ahead in every same-day comparison. Measuring the noise first is what makes the null result on the LLM believable.

## List 1: top 3 improvements to StochasticGoose itself

Ranked by expected gain per day of work, on top of `mb_gated_att`. None needs an LLM, and each goes behind a new flag so A0 stays unchanged.

### 1. Give Goose its missing actions: ACTION7 (undo) and a reset when the screen freezes

- **What.** Add ACTION7 as a sixth button on the six games that advertise it. Add a watchdog: after about 300 decisions with no change to the masked screen, send RESET.
- **Why.** `run_local.py` filters ACTION7 out, so ar25, bp35, lf52, sb26, sk48 and su15 play a reduced action space. Two of them are the "frozen" games: lf52 saw 18 to 21 distinct screens in 100,000 actions and sb26 saw 11 (Plan B and return-map confirms). The agent sends RESET only after GAME_OVER, so it cannot leave a dead screen. [ARC's docs](https://docs.arcprize.org/actions) define ACTION7 as undo, and [Franzen](https://github.com/da-fr/arc-agi-3-solution) reports that exposing it helped substantially.
- **Pros.** About one day. It removes a limitation the report must otherwise disclose. The test is low-noise: A0 scores 17 levels in 18 runs on these six games in both of its sweeps.
- **Cons.** Undo returns to a seen screen, so the novelty label will teach Goose to avoid it. It is an escape, not a search tool. A sixth output needs a new preset. Two RESETs in a row restart the whole game.
- **Test.** Count frozen runs in existing recordings first. Then 6 games x 3 seeds x 2 arms, 6 to 7 GPU-hours.

### 2. Carry what transfers between levels inside Goose

- **What.** Move S5's two LLM-free parts into `upgrades.py`: carried colour no-ops and goal hints. Extend hints to button games with a label bonus when a move changes a hint-colour cell. Add a zero-code arm: `mb_gated_att` with `EVAL_RESET_ON_LEVEL=0`.
- **Why.** It is the only level-to-level transfer with positive evidence. Carried no-ops would have covered 60.1% of vc33's and 22.9% of ft09's moves in each new level's first 2,000 actions. All three sweeps favoured the guide (+1, +5, +9 levels). Keeping the network across levels scored 18 vs 14 levels in upgrade screen 1 and was never tried on the adopted agent.
- **Pros.** The click version is built and tested. It credits the explorer with what the data assigns to it. It should help time-to-level most.
- **Cons.** Each gain is inside its noise band. In G4, 33 of 86 hints were dropped as ineffective. Persistence made ft09 about 18 times more redundant under the old label. There is no LLM in it.
- **Test.** The hint-active games (s5i5, r11l, lf52, sp80, vc33, cd82) x 4 seeds x 2 arms, about 9 GPU-hours.

### 3. Fix what Goose sees: animation frames and the ends of bars

- **What.** (a) Count a move as a change when any intermediate frame differs, with novelty credit for an unseen animation. (b) Run each detected bar to the screen edge in the agent's own mask, as `gb/s0_mask.py` does offline.
- **Why.** Goose keeps only the last frame of each animation. Franzen found some games close to unsolvable without the others and names tn36, where A0 has one level in six runs. On bars: after they were masked, dc22's abandoned map walks fell from 881 to 67 and the adopted agent took dc22 from 0 to 7 levels. The agent's mask still lacks a bar's last cells.
- **Pros.** Small and LLM-free. Part (b) reuses existing code and can be checked on recordings before any GPU run.
- **Cons.** The most speculative of the three. Recordings keep only final frames, so (a) cannot be checked offline. A missing bar end inflates the state count by a bounded amount, so (b) may gain little.
- **Test.** A 2,000-action probe per game to find which games animate, then those games x 3 seeds.

## List 2: top 3 updates to the current Glassbox implementation

All three keep the idea statement of 2 Nov. They are ranked by what they buy for the report within a week.

### 1. Swap the rule writer for a stronger model: same pipeline, same gate

- **What.** Point `gb/core/llm.py` at an OpenAI-compatible API and rerun S2 to S4 offline on the 8 dev games with Qwen3.8-Flash, the hosted Flash-Next. Add one frontier model on 2 or 3 games as a ceiling.
- **Why.** The plan names the bottleneck as "the model doesn't write enough exact rules". Outside, the same harness went from 1.21% to 27.89% when its 27B gave way to Flash-Next. This is the cheapest test of "is it the pipeline or the model?".
- **Pros.** Half a day to a day: the client already speaks the OpenAI protocol and caches answers. No GPU. G1 generated 6.27M tokens, so I estimate under $15 at [$0.15 / $0.47 per million](https://openrouter.ai/compare/qwen/qwen3-coder-next/qwen/qwen3.8-flash). Flash-Next is open-weights.
- **Cons.** Better rules alone add no levels, because online they still only prune. The model postdates the public games, so label it a comparison arm. The API may reject `n > 1` or `chat_template_kwargs`.
- **Test.** G1's table for both writers. If mean cell coverage at least doubles (32% to 64%), the model was the limit.

### 2. Give the book a goal: an LLM-written, replay-checked progress test

- **What.** A new S2 task: show the writer the frames before each recorded level-up and ask for `progress(board, p)` in H1's primitives. S3 keeps it only if it ranks the board before a held-out win above earlier boards more often than the two-colour hint does. S5 uses it in place of the hint, and as the map's stall target.
- **Why.** Goal inference is what the "model plays" harnesses add and what Glassbox lacks. The one online gain came from a crude LLM-free hint. Tycho's authors make the same point: matching transitions does not show the objective was found.
- **Pros.** It gives the LLM a job that can add levels, not only save moves. It reuses S2, S3, re-binding and the hint hook, and it fits the idea statement.
- **Cons.** The closest earlier tries failed: Probe C never beat a one-line rule and Coach ranked at chance. So use the stronger writer from item 1. A wrong goal misleads: [another team's agent](https://github.com/pcurran100/arc-agi-3) took the move counter for the goal.
- **Cost and test.** Two to three days. The offline check comes first and needs no GPU. Then 8 dev games x 4 seeds with a same-day A0, about 12 GPU-hours.

### 3. Score what the guide changes, and finish the ablation

- **What.** (a) Re-score the 150 G4 runs and the 64 dev-sweep-2 runs on time-to-level: levels by 2,000, 10,000 and 30,000 actions, and AULC. (b) Run the missing LLM-free arm on the 6 games that have rules. (c) Test new ideas on a targeted game set with 4 to 6 seeds.
- **Why.** The guide acts in the first 2,000 moves of each new level, so its effect should be earlier level-ups, which were never scored. G4 has no LLM-free arm, so the LLM's share has no direct number. With about 8 levels of noise per 75-run total, only large effects pass a level-count confirm.
- **Pros.** Part (a) takes hours and no GPU; `analyze_curves.py` already builds the curves. Part (b) is 18 runs, about 3.5 GPU-hours. The report needs both tables anyway.
- **Cons.** A metric chosen after the runs is exploratory and must be labelled so. It changes no agent, so by itself it adds no levels.

## List 3: top 3 completely new ideas

Ranked by expected payoff, with a note on what is buildable by 17 Nov. All three change what the LLM does, so check them against the 2 Nov idea lock before building. As reference arms or future work they are safe either way.

### 1. Let the LLM play first, and let Goose take over when it stalls

- **What.** Run the open-source Duck harness on the local engine ([Tufa Labs' original](https://tufalabs.ai/research/duck-harness/); [Franzen's version](https://github.com/da-fr/arc-agi-3-solution), Apache-2.0). It is an LLM in a Python REPL that inspects the board, tests mechanics and sends actions. Cap it per game, save its action list, replay it as a prefix in `run_local.py`, then let `mb_gated_att` continue to 100,000 actions.
- **Why.** This is the setup behind every large published gain: 27.89% on hidden games, and about 45 averaged over 4 passes on the public games. It uses the LLM for goal inference and planning. Any level the model clears is a level Goose starts beyond.
- **Model.** The 27B fits the 5090, but expect low single digits: the harness scored 1.60 with Qwen 3.6 27B. Flash-Next needs about 74 GB, so use the hosted version. I estimate $5 to $15 per 25-game pass; measure it on two games first.
- **Pros.** Highest expected payoff in this document. The code exists and runs on the same engine. It answers "LLM as player vs LLM as rule writer" on the same games.
- **Cons.** It is a different idea from the locked statement, and other people's harness: our part would be the relay and the comparison. Integration risk on a 32 GB card. Newer models may have seen the public games.
- **By 17 Nov.** A pilot: 8 dev games, one or two seeds, both models, in two to three days. Not a 25 x 3 confirm.

### 2. Ask the LLM for plans when Goose stalls, and let the game check them

- **What.** When Goose stalls, pause and send the model the level's first and current screens, S1's object facts, what each action did, and the replay of the last win. Ask for up to K short plans, such as "click every red tile once". Run each for real. A plan that wins ends the stall; one that finds new screens feeds the map.
- **Why.** At 100,000 actions per game, actions are nearly free and LLM calls are scarce, the reverse of the RHAE setting. So spend calls on a few guesses and let the game verify them. No exact world model is needed. It is Coach with images, plans and execution in place of a text summary, single-move rankings and offline scoring.
- **Pros.** Small: the stall signal, map, S1 facts and LLM client exist. Ten to thirty calls per game. The placebo is easy: random plans on the same schedule.
- **Cons.** Coach scored at chance with the 27B (21 hits against 21.2 expected), so it needs the stronger model. K blind plans will not hit a long exact sequence.
- **By 17 Nov.** A pilot in three to four days. It competes with idea 1, which ranks higher because there the model can inspect and iterate.

### 3. Train a small prior on solved-level traces and carry it across games

- **What.** Collect (screen, action) pairs from solved levels, cut to the shortest known path. Train a small network to predict the next winning action: Goose's CNN with an extra head, or a 2 to 9B vision model with LoRA on Colab. Use it as a prior on Goose's sampling in new games, tested leave-one-game-out.
- **Why.** It is the only idea here in which something learned carries from game to game. That is the project's title question and the literal reading of "a small, self-trained model". Hand-carried rules did not transfer: 0 of 792 template tries.
- **Pros.** It uses data the project has, over 100 level-ups per 75-run sweep. The held-out tiers give it a clean protocol. It is reportable even as a negative result.
- **Cons.** Little data, dominated by levels 1 and 2. Games are built to be novel, so negative transfer is likely. One to two weeks to do properly.
- **By 17 Nov.** No. A two-day offline pilot is possible; otherwise it belongs in future work.

## Suggested order for the week

This order does the free steps first and keeps the GPU and the API busy at the same time. It fits about five days of work and three overnight runs before the freeze on 17 Nov 2026.

1. **Tue to Wed, no GPU.** Re-score G4 and dev sweep 2 on time-to-level (List 2, item 3a). Count frozen runs in the 150 G4 recordings. Probe which games return multi-frame animations. These three numbers decide how much the List 1 items deserve.
2. **Wed to Thu.** Build ACTION7 and the reset watchdog (List 1, item 1). Overnight: 6 games x 3 seeds x 2 arms.
3. **Thu to Fri.** Add the API backend and run the writer swap offline (List 2, item 1). It needs no GPU, so run the LLM-free arm on the 6 rule games the same night (List 2, item 3b).
4. **Fri to Sun.** Pick one. If the writer swap at least doubles coverage, build the goal test (List 2, item 2). Otherwise run the "LLM plays" pilot on the dev games (List 3, item 1), after asking the instructor whether a reference arm is allowed under the idea lock.
5. **Mon to Tue.** One dev sweep with a same-day A0 for whichever of steps 2 and 4 looked best. Then freeze and write.

Effort and cost for all nine ideas. GPU-hours assume four runs at a time; API costs are my estimates at $0.15 / $0.47 per million tokens.

| Idea | Work | GPU-hours | API cost | By 17 Nov |
| --- | --- | --- | --- | --- |
| List 1.1 ACTION7 and reset watchdog | 1 day | 6 to 7 | none | yes |
| List 1.2 Carry between levels inside Goose | 1 to 2 days | about 9 | none | yes |
| List 1.3 Animation frames and bar ends | 1 to 2 days | 4 to 8 | none | yes |
| List 2.1 Stronger rule writer | 0.5 to 1 day | none | under $15, plus $20 to $45 for a frontier model on 2 to 3 games | yes |
| List 2.2 LLM-written goal test | 2 to 3 days | about 12 | under $10 | offline check yes; dev sweep is tight |
| List 2.3 Time-to-level re-score and LLM-free arm | 0.5 day | 3.5 | none | yes |
| List 3.1 LLM plays, Goose takes over | 2 to 3 days | 4 to 8 for the 27B arm | $10 to $40 | pilot only |
| List 3.2 LLM plans at stalls | 3 to 4 days | about 12 | under $20 | pilot only |
| List 3.3 Learned prior across games | 1 to 2 weeks | Colab | none | no |

For the report, due 7 Dec 2026, whatever gets built:

- Lead with the explorer (54 to 79 to 112 levels) and its mechanisms.
- State G4 with its noise, and say plainly that the gain belongs to the LLM-free guide.
- Add a comparison section built on the two tables in "How our setup differs", and name the two regimes: RHAE at human-scale budgets, and levels at 100,000 actions.
- Present the LLM track as a measured study of where a local 27B rule writer breaks: yield, carry and goal. The writer swap is the controlled test of the first.
- Cite the independent team that hit the same wall with the same design.

## Sources

Outside sources, all read on 10 Nov 2026:

- [ARC-AGI-3 technical report](https://arxiv.org/html/2603.24621v1) (arXiv 2603.24621): the RHAE formula, the 5x cutoff, preview results, launch scores.
- [ARC Prize leaderboard](https://arcprize.org/leaderboard), [GPT-6 Astra on ARC-AGI-3](https://arcprize.org/blog/astra), [Milestone Prize #1](https://arcprize.org/blog/arc-prize-2026-milestone-1), [Actions documentation](https://docs.arcprize.org/actions).
- [Tufa Labs, Duck harness](https://tufalabs.ai/research/duck-harness/): 1.60 +/- 0.45 on the public games with Qwen 3.6 27B.
- [Daniel Franzen, Milestone 2 solution and write-up](https://github.com/da-fr/arc-agi-3-solution): what helped (stronger models, animations, UNDO, larger context) and what did not.
- [Rodionov, Executable World Models](https://arxiv.org/abs/2605.05138) and the [component study](https://arxiv.org/abs/2607.15439); [Tycho](https://arxiv.org/abs/2607.28287); [OPINE-World](https://arxiv.org/abs/2607.01531).
- Competition repos: [pcurran100](https://github.com/pcurran100/arc-agi-3) (graph agent, then a 27B world-model agent), [hammadfarooq-ai](https://github.com/hammadfarooq-ai/arc-prize-2026-arc-agi-3) (graph explorer, then measured harness changes), [veeda241](https://github.com/veeda241/ARC-Prize-2026---ARC-AGI-3), [Sahasawatt](https://github.com/Sahasawatt/arc-agi-3-agent).
- [OpenRouter, Qwen3.8 Flash pricing](https://openrouter.ai/compare/qwen/qwen3-coder-next/qwen/qwen3.8-flash).
- Kaggle pages need JavaScript, so I read them through the search index only: the [public leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard) (75.56), the [Milestone 2 comparison](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792) (27.89; 73.6 GB of weights on the GPU) and [Tufa Labs' Milestone 1 write-up](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133) (1.21%). Check these three before quoting them in the report.

Repo files, branch `glassbox` at `c9f59df`:

- Status and limits: `CLAUDE.md`, `docs/README.md`.
- Explorer results: `docs/1-goose-upgrades/` (Plan B, return map, upgrade screens and confirm); per-game levels from project doc `claude/project-review-2026-10-13.md`, Appendix B.
- Glassbox results: `docs/3-glassbox/` (`g1-offline-rules.md`, `g2-online.md`, `g3-templates.md`, `carry-later-levels.md`, `g4-confirm.md`, `glassbox-plan.md`).
- Code read: `custom_agents/action.py`, `upgrades.py`, `return_map.py`, `gb/s5_act.py`, `gb/s0_mask.py`, `gb/core/llm.py`, `run_local.py`.

The connected WSL folder could not be mounted, so raw run outputs under `results/` were not read. Every repo number here is as published in the repo's own write-ups.
