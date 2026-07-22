#!/usr/bin/env python3
"""run_local.py - drive the StochasticGoose agent against the LOCAL engine.

Sibling to the API path (`make action`), which is left completely untouched.
Reuses the agent brain in custom_agents/action.py as-is and the eval_common
contract, but owns the game loop and points it at the in-process arc_agi engine
instead of the hosted API - removing the ~50 ms/action HTTP round trip.

DESIGN NOTE (why this doesn't touch the harness package):
  Importing anything from ARC-AGI-3-Agents/agents runs its __init__.py, which
  eagerly imports optional templates (LangGraph -> broken Pillow ref) and
  `custom_agent`. That eager chain is fragile. So we DON'T import that package.
  Instead we synthesize a minimal `agents` package in sys.modules from only what
  action.py needs:
     * agents.structs      -> the REAL module (GameAction/GameState/FrameData)
     * agents.agent.Agent  -> a lightweight stand-in base; Action subclasses it
                              and only uses self.game_id + self.action_counter.
  The real game loop lives in main() below, so we never need the harness's
  networked base class or its play loop.

Run (smoke test):
  EVAL_SEED=0 EVAL_MAX_ACTIONS=10000 PYTHONHASHSEED=0 \
      uv run python run_local.py --game ls20

Then analyze the corpus it prints the path to:
  uv run python compute_metrics.py <that path> --game ls20 --agent goose \
      --seed 0 --suite suite_summary.csv
"""
import argparse
import importlib.util
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(ROOT, "ARC-AGI-3-Agents")
for _p in (HARNESS, os.path.join(ROOT, "custom_agents"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _install_minimal_agents_pkg():
    """Put a minimal `agents` package in sys.modules so importing action.py
    never triggers the real (fragile) agents/__init__.py."""
    if "agents" in sys.modules:
        return

    pkg = types.ModuleType("agents")
    pkg.__path__ = [os.path.join(HARNESS, "agents")]   # real dir for any relative import
    pkg.__package__ = "agents"
    sys.modules["agents"] = pkg

    # Load the REAL structs module (leaf: enums + dataclasses) by file path.
    structs_path = os.path.join(HARNESS, "agents", "structs.py")
    spec = importlib.util.spec_from_file_location("agents.structs", structs_path)
    structs = importlib.util.module_from_spec(spec)
    sys.modules["agents.structs"] = structs
    spec.loader.exec_module(structs)
    pkg.structs = structs

    # Provide a lightweight stand-in base Agent. The real base owns networking
    # and the play loop, neither of which we use; Action only reads game_id and
    # action_counter off the base.
    agent_mod = types.ModuleType("agents.agent")

    class Agent:
        def __init__(self, *args, **kwargs):
            self.game_id = kwargs.get("game_id") or (args[0] if args else None)
            self.action_counter = 0

    agent_mod.Agent = Agent
    sys.modules["agents.agent"] = agent_mod
    pkg.agent = agent_mod


_install_minimal_agents_pkg()

import arc_agi
from arcengine import GameAction as EAction        # engine action enum
from agents.structs import GameAction as HAction   # harness enums (what the agent
from agents.structs import GameState as HState      # expects to see)
from eval_common import resolve_seed, resolve_max_actions

GRID = 64


def _extract_grid(obs):
    """Pull the frame grid off the engine observation (FrameDataRaw)."""
    for attr in ("frame", "frames", "grid"):
        g = getattr(obs, attr, None)
        if g is not None:
            return g
    raise AttributeError("no grid on engine obs (tried .frame/.frames/.grid) - "
                         "inspect the engine observation and add the attr here")


def _extract_state(obs):
    st = getattr(obs, "state", None)
    if st is None:
        raise AttributeError("engine obs has no .state")
    return HState[st.name]        # engine GameState -> harness GameState by name


def _extract_score(obs):
    s = getattr(obs, "score", None)
    if s is not None:
        return s
    return getattr(obs, "levels_completed", 0)


class ShimFrame:
    """Minimal stand-in for the harness FrameData. Exposes exactly the four
    attributes Action.choose_action / is_done read, using the HARNESS enums so
    `state is GameState.WIN` and `action.value` behave like the API path."""
    __slots__ = ("frame", "score", "state", "available_actions")

    def __init__(self, obs, action_space):
        self.frame = _extract_grid(obs)
        self.score = _extract_score(obs)
        self.state = _extract_state(obs)
        self.available_actions = [HAction[a.name] for a in (action_space or [])]


def to_engine_action(haction, prev_action_idx):
    """Map the agent's chosen action to (engine_action, data_dict, is_reset).

    Prefers the agent's own unified index (set at the end of choose_action):
      0-4  -> ACTION1-5 ;  5 + (64*y + x) -> ACTION6 click at (x, y)."""
    if haction.name == "RESET":
        return None, {}, True
    if prev_action_idx is not None:
        if prev_action_idx < 5:
            return getattr(EAction, f"ACTION{prev_action_idx + 1}"), {}, False
        coord = prev_action_idx - 5
        y, x = divmod(coord, GRID)
        return EAction.ACTION6, {"x": int(x), "y": int(y)}, False
    # weird-frame fallback path in the agent: a plain button by name
    return getattr(EAction, haction.name), {}, False


def build_agent(game_id):
    """Import and instantiate the Action agent. Its __init__ does all the real
    brain/logger/corpus setup; our stand-in base supplies game_id + counter."""
    from action import Action
    return Action(game_id=game_id)


def make_arcade(offline):
    """Construct the Arcade, tolerant of signature differences across engine
    versions (operation_mode may or may not be accepted)."""
    try:
        from arc_agi import OperationMode
        mode = OperationMode.OFFLINE if offline else OperationMode.NORMAL
        return arc_agi.Arcade(operation_mode=mode)
    except (ImportError, TypeError, AttributeError):
        return arc_agi.Arcade()


def make_env(arc, game_id, game_seed, render):
    """Make the env, tolerant of which kwargs this engine build accepts."""
    kwargs = {"seed": game_seed}
    if render:
        kwargs["render_mode"] = render
    try:
        return arc.make(game_id, **kwargs)
    except TypeError:
        return arc.make(game_id)      # older/newer make() without these kwargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--offline", action="store_true",
                    help="OperationMode.OFFLINE (needs local game files); default "
                         "plays locally + syncs scorecard via API")
    ap.add_argument("--render", default=None, choices=[None, "terminal", "human"])
    args = ap.parse_args()

    seed, seed_source = resolve_seed(args.game)
    cap = resolve_max_actions()
    game_seed = seed % (2 ** 31 - 1)

    agent = build_agent(args.game)
    print(f"[run_local] game={args.game} seed={seed} ({seed_source}) "
          f"cap={cap} device={getattr(agent, 'device', '?')}")

    arc = make_arcade(args.offline)
    env = make_env(arc, args.game, game_seed, args.render)
    if env is None:
        raise SystemExit(f"failed to make env for {args.game}")

    obs = env.reset()
    frame = ShimFrame(obs, getattr(env, "action_space", None))

    t0 = time.time()
    consecutive_resets = 0
    while agent.action_counter < cap:
        if agent.is_done([frame], frame):
            print(f"[run_local] is_done at {agent.action_counter} "
                  f"(state={frame.state.name})")
            break

        action = agent.choose_action([frame], frame)
        eng_action, data, is_reset = to_engine_action(
            action, getattr(agent, "prev_action_idx", None))

        if is_reset:
            obs = env.reset()
            consecutive_resets += 1
            if consecutive_resets > 10:
                raise SystemExit("10+ resets with no progress - check reset "
                                 "semantics / initial state handling")
        else:
            obs = env.step(eng_action, data=data)
            consecutive_resets = 0

        frame = ShimFrame(obs, getattr(env, "action_space", None))
        agent.action_counter += 1

        if agent.action_counter % 1000 == 0:
            aps = agent.action_counter / (time.time() - t0)
            print(f"  {agent.action_counter:>7} actions  score={frame.score}  "
                  f"{aps:5.1f} act/s")

    if getattr(agent, "transition_logger", None) is not None:
        agent.transition_logger.flush()
    try:
        agent.writer.flush()
    except Exception:
        pass

    dt = time.time() - t0
    print(f"[run_local] done: {agent.action_counter} actions in {dt:.1f}s "
          f"({agent.action_counter / max(dt, 1e-9):.1f} act/s)")
    try:
        sc = arc.get_scorecard()
        if sc is not None:
            print(f"[run_local] scorecard score={getattr(sc, 'score', sc)}")
    except Exception as e:
        print(f"[run_local] (scorecard unavailable: {e})")
    print(f"[run_local] transitions: "
          f"{os.path.join(getattr(agent, 'log_dir', '.'), 'transitions')}")


if __name__ == "__main__":
    main()
