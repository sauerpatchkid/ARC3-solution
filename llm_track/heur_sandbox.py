"""heur_sandbox.py — check and run Qwen-written heuristic code.

Two layers, in order:

  validate(src)   STATIC. The code must parse, define score(board, api), and use
                  no imports, no attribute starting with "_" (blocks the classic
                  ().__class__ escapes), and nothing from a denylist of file,
                  process, introspection and numpy-I/O names.
  evaluate(...)   DYNAMIC, in a CHILD PROCESS with restricted builtins and a
                  wall-clock timeout, so a crash, an infinite loop or a memory
                  blow-up cannot take down the caller. It runs the heuristic on
                  the referee's recorded boards and returns, per board, the
                  percentile of the move that was actually played, plus timings.

Then the gates (see GATES): average call time, and "not constant everywhere".
Only code that passes all of them is graded.

This guards against mistakes and accidents in locally generated code; it is not
a hardened security boundary, and does not need to be for code produced by a
local model on the owner's machine.
"""
import ast
import builtins
import multiprocessing as mp
import queue
import time

import numpy as np

GATES = {
    "max_mean_ms": 1.0,          # keeps Goose at >= ~90% of its speed
    "max_constant_frac": 0.90,   # constant on more boards than this = no opinion
}

_SAFE = ("abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
         "int", "isinstance", "len", "list", "map", "max", "min", "range",
         "reversed", "round", "set", "sorted", "sum", "tuple", "zip",
         "ValueError", "Exception")
SAFE_BUILTINS = {n: getattr(builtins, n) for n in _SAFE}


def _numpy_only_import(name, globals=None, locals=None, fromlist=(), level=0):
    """numpy's C code imports its own submodules on demand (for example to build
    an error message), looking up __import__ in the CALLER's builtins - which
    here are ours. Without this, a plain IndexError in a heuristic surfaced as
    "KeyError: '__import__'", and numpy functions that load a submodule on first
    use failed outright. Heuristic source still cannot contain an import: the
    static check rejects Import nodes and the name __import__."""
    if name == "numpy" or name.startswith("numpy."):
        return builtins.__import__(name, globals, locals, fromlist, level)
    raise ImportError(f"import of {name!r} is not allowed")


SAFE_BUILTINS["__import__"] = _numpy_only_import

DENY_NAMES = {"__import__", "eval", "exec", "compile", "open", "input", "globals",
              "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
              "exit", "quit", "help", "memoryview", "type", "object", "super",
              "dir", "id", "print"}
DENY_ATTRS = {"load", "save", "savez", "savez_compressed", "fromfile", "tofile",
              "loadtxt", "savetxt", "genfromtxt", "memmap", "lib", "ctypeslib",
              "f2py", "testing", "distutils", "os", "sys", "ctypes", "frombuffer",
              "DataSource", "fromregex"}
DENY_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
              ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.With,
              ast.AsyncWith, ast.Delete)
TOP_LEVEL_OK = (ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Expr)


def validate(src):
    """Static check. Returns (ok, reason)."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg} (line {e.lineno})"
    for node in tree.body:
        if not isinstance(node, TOP_LEVEL_OK):
            return False, f"only functions and constants allowed at top level, got {type(node).__name__}"
    for node in ast.walk(tree):
        if isinstance(node, DENY_NODES):
            return False, f"{type(node).__name__} is not allowed"
        if isinstance(node, ast.Name) and node.id in DENY_NAMES:
            return False, f"name '{node.id}' is not allowed"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return False, f"attribute '{node.attr}' is not allowed"
            if node.attr in DENY_ATTRS:
                return False, f"attribute '{node.attr}' is not allowed"
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "score"]
    if not fn:
        return False, "no function named score"
    if len(fn[0].args.args) != 2:
        return False, "score must take exactly (board, api)"
    return True, "ok"


def compile_heuristic(src):
    """Validated source -> (score function, IDEA string)."""
    ok, why = validate(src)
    if not ok:
        raise ValueError(why)
    ns = {"np": np, "__builtins__": SAFE_BUILTINS}
    exec(compile(ast.parse(src), "<heuristic>", "exec"), ns)
    return ns["score"], str(ns.get("IDEA", ""))


def check_output(out):
    """(buttons, clicks) -> float arrays of shape (5,) and (64, 64), or raise."""
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise ValueError("score must return (buttons, clicks)")
    b = np.asarray(out[0], dtype=float).reshape(-1)
    c = np.asarray(out[1], dtype=float)
    if b.shape != (5,):
        raise ValueError(f"buttons must have 5 values, got shape {b.shape}")
    if c.shape != (64, 64):
        raise ValueError(f"clicks must be 64x64, got shape {c.shape}")
    if not (np.isfinite(b).all() and np.isfinite(c).all()):
        raise ValueError("scores must be finite (no NaN / inf)")
    return b, c


def _worker(src, referee_dir, where, max_samples, seed, q):
    """Child process: run the heuristic over the referee's boards."""
    try:
        from .heur_referee import Referee
        fn, idea = compile_heuristic(src)
        ref = Referee(referee_dir)
        idx = ref.select(**where)
        if max_samples and len(idx) > max_samples:
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_samples, replace=False))
        pct = np.empty(len(idx)); ms = np.empty(len(idx)); const = 0
        for k, i in enumerate(idx):
            board, api = ref.inputs(i)
            t0 = time.perf_counter()
            out = fn(board, api)
            ms[k] = (time.perf_counter() - t0) * 1000.0
            b, c = check_output(out)
            pct[k], is_const = ref.taken_percentile(i, b, c)
            const += is_const
        q.put({"ok": True, "idea": idea, "idx": idx, "pct": pct, "ms": ms,
               "constant_frac": const / max(len(idx), 1)})
    except Exception as e:                       # report, never raise across the pipe
        q.put({"ok": False, "stage": "runtime", "reason": f"{type(e).__name__}: {e}"})


def evaluate(src, referee_dir, where=None, timeout_s=300, max_samples=None, seed=0,
             return_samples=False):
    """Validate, run in a child process, apply the gates, and grade.

    Returns a dict with ok, stage ('static' | 'runtime' | 'timeout' | 'gate' |
    'graded'), reason, and for graded code: auc, ci, n, n_events, ms_mean,
    ms_p95, constant_frac, idea. With return_samples, also `idx` and `pct` (the
    played move's percentile for each referee sample), which the writer uses to
    show Qwen the moves its heuristic got most wrong.
    """
    ok, why = validate(src)
    if not ok:
        return {"ok": False, "stage": "static", "reason": why}
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(src, referee_dir, where or {}, max_samples, seed, q))
    p.start()
    # Poll rather than block: a worker that dies without reporting (segfault,
    # out-of-memory kill, a failed spawn) must fail fast, not after timeout_s.
    res, deadline = None, time.time() + timeout_s
    while res is None and time.time() < deadline:
        try:
            res = q.get(timeout=1.0)
        except queue.Empty:
            if not p.is_alive():
                try:
                    res = q.get(timeout=1.0)          # it may have reported just before exiting
                except queue.Empty:
                    return {"ok": False, "stage": "runtime",
                            "reason": f"worker process exited (code {p.exitcode}) without a result"}
    if res is None:
        p.terminate(); p.join(5)
        return {"ok": False, "stage": "timeout", "reason": f"no result within {timeout_s}s"}
    p.join(10)
    if p.is_alive():
        p.terminate()
    if not res["ok"]:
        return res

    ms_mean = float(res["ms"].mean()) if len(res["ms"]) else 0.0
    out = {"idea": res["idea"], "ms_mean": round(ms_mean, 3),
           "ms_p95": round(float(np.percentile(res["ms"], 95)), 3) if len(res["ms"]) else 0.0,
           "constant_frac": round(res["constant_frac"], 3)}
    if ms_mean > GATES["max_mean_ms"]:
        return {**out, "ok": False, "stage": "gate",
                "reason": f"too slow: {ms_mean:.2f} ms per call (limit {GATES['max_mean_ms']})"}
    if res["constant_frac"] > GATES["max_constant_frac"]:
        return {**out, "ok": False, "stage": "gate",
                "reason": f"same score for every option on {res['constant_frac']:.0%} of boards"}

    from .heur_referee import Referee
    grade = Referee(referee_dir, light=True).grade(res["idx"], res["pct"])
    extra = {"idx": res["idx"], "pct": res["pct"]} if return_samples else {}
    return {**out, **grade, **extra, "ok": True, "stage": "graded", "reason": "ok"}
