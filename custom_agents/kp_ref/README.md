# kp_ref: the groupmate's planner, as received. Do not edit.

These files are the reference for the Knowledge Planner
(`docs/4-knowledge-planner/kp-plan.md`). They are a Team B groupmate's work,
not this repo's: the agents E18, E19 and E20 of the "Working Results" report
(5 Nov 2026), which are one method built in three layers, and E21, the
explorer the same report adds for games played with keys.

| File | Lines | sha256 |
|---|---|---|
| `e18_knowledge_planner.py` | 1,194 | `bc48c989e7431c40e59ba2511870b1d89f57171bf5a17e7c9a48623a94605cd4` |
| `e19_adaptive_planner.py` | 333 | `fdf5b446434efaa4b824e646fb45033efcbab111a68987f42e8834e6fffeac83` |
| `e20_learned_knowledge.py` | 506 | `fb370914aed5b25f579adb37479ea3421e5c7452e97ac2accac1b8d10d0b6836` |
| `knowledge_bank.py` | 183 | `77a917efeb260de50c7c205c263b27a1105033c8209f7e1861781dd496dfa280` |
| `e21_object_explorer.py` (added 20 Nov) | 483 | `76b262c953c603399b3fa4522c965e577e273813e9ecc9616c3410f2750d5be9` |

Copied byte for byte on 19 Nov 2026 from `Final_Working_Results_Nov16.zip`
(the groupmate's hand-in, downloaded 16 Nov; the files inside are dated 4 and
5 Nov). The first four hashes are the ones in the plan's Appendix C. E21 was
copied from the same archive on 20 Nov. Check them with:

```bash
sha256sum custom_agents/kp_ref/*.py
```

**Why they are never edited.** Every result of the Knowledge Planner is a
claim about "the groupmate's E20, unmodified". A changed byte here would make
that false without anyone seeing it. Whatever has to behave differently is
done from outside, at run time: `experiments/kp/run_reference.py` and, later,
`custom_agents/kp_bridge.py` replace a function on the loaded module and leave
the files alone.

**How they are loaded.** Nothing imports this folder as a package. The loader
puts it on `sys.path` and imports the modules by name. The files import
`pandas` at the top and use it only in their own `main()`; where pandas is not
installed, the loader registers an empty stand-in under that name, so no
dependency is added to the repo.

**What they need:** `torch`, `numpy`, `arc_agi`, `arcengine` (all already used
by the baseline).

`knowledge_bank.py` is kept with the others; nothing before the plan's
optional Phase 6 uses it.

**E21 (`e21_object_explorer.py`)** is loaded only when the planner's settings
ask for it (`[explore] keys = "e21"`, off in `experiments/kp/kp.toml`). It is
a subclass of E20: on a game that offers only clicks it calls E20 unchanged,
and on a game with keys it searches the screens systematically and carries
what a cleared level taught to the next. The same archive holds E22 (a planner
on top of E21) and E24 (another explorer); a check on recordings found both
worse than E21 behind this project's Goose, so they are not here
(`docs/4-knowledge-planner/kp-explorer.md`).
