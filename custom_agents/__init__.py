"""Agent registry — this is where you plug in a new agent.

ADDING YOUR AGENT (three steps, no other file needs to change)

  1. Put your file in this directory, e.g. custom_agents/my_agent.py.
     Copy custom_agents/TEMPLATE.py to start; it documents the surface the
     runner drives and wires up the eval contract for you.

  2. Add one line to REGISTRY below:
          The key is the name you pass to --agent; the value is
     (module name inside this package, class name).

  3. Run it:
         uv run python run_local.py --game ft09 --agent mine
         make bench SUITE=standard AGENT=mine

Everything else — seeding, the action cap, the transition corpus, metrics,
the benchmark suites, the comparison table — works for your agent the moment
it is registered, because they all go through eval_common and the corpus
rather than through anything agent-specific.

WHY THE VALUES ARE STRINGS. Agent modules import `agents.structs`, which is
the harness package that run_local.py synthesizes into sys.modules at startup.
Importing an agent at module-import time would therefore fail. Keeping the
registry as plain data means this file is safe to import from anywhere, and
the actual import happens in load_agent(), after the runner is ready.
"""
import importlib

# name -> (module inside custom_agents/, class name)
REGISTRY = {
    "goose": ("action", "Action"),
    "random": ("random_agent", "RandomAgent"),
}


def available():
    """Registered agent names, sorted — used for --agent's choices."""
    return sorted(REGISTRY)


def load_agent(name, game_id):
    """Instantiate a registered agent for one game.

    The agent's own __init__ does the real work (model, logging, corpus); the
    runner only supplies game_id.
    """
    if name not in REGISTRY:
        raise SystemExit(
            f"unknown agent {name!r}. Registered: {', '.join(available())}.\n"
            f"To add one, see the instructions at the top of "
            f"custom_agents/__init__.py")
    module_name, class_name = REGISTRY[name]
    # Imported as a bare module name: run_local.py puts this directory on
    # sys.path, which is also how the agent modules find each other
    # (action.py does `from view_utils import ...`).
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(game_id=game_id)
