"""sandbox.py - check and run LLM-written rule code.

Two layers, in order:
  validate(src)   STATIC. The code must parse, define the required functions,
                  and use no imports, no attribute starting with "_" (blocks the
                  classic ().__class__ escapes), and nothing from a denylist of
                  file, process, introspection and numpy-I/O names. Denied names
                  are denied as attributes too: numpy's own modules import
                  `builtins`, so np.ma.core.builtins.open reached the real open().
  run_in_child    DYNAMIC, in a spawned CHILD PROCESS with restricted builtins
                  and a wall-clock timeout, so a crash, an infinite loop or a
                  memory blow-up cannot take down the caller. The worker gets
                  board arrays and the `api` object only - never engine objects.

This guards against mistakes and accidents in locally generated code. It is not
a hardened security boundary (docs/plans/llm-rulebook.md section 3.3 says what
the report may claim). Copied from legacy/llm_track/heur_sandbox.py.
"""
import ast
import builtins
import multiprocessing as mp
import queue
import time

import numpy as np

RULE_FUNCS = {"applies": 3, "predict": 3}

_SAFE = ("abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
         "int", "isinstance", "len", "list", "map", "max", "min", "range",
         "reversed", "round", "set", "sorted", "sum", "tuple", "zip",
         "ValueError", "Exception")
SAFE_BUILTINS = {n: getattr(builtins, n) for n in _SAFE}


def _numpy_only_import(name, globals=None, locals=None, fromlist=(), level=0):
    """numpy's C code imports its own submodules on demand (for example to build
    an error message), looking up __import__ in the CALLER's builtins - which
    here are ours. Rule source still cannot contain an import: the static check
    rejects Import nodes and the name __import__."""
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
              "DataSource", "fromregex", "builtins"} | DENY_NAMES
DENY_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
              ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.With,
              ast.AsyncWith, ast.Delete)
TOP_LEVEL_OK = (ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Expr)


def validate(src, required=RULE_FUNCS):
    """Static check. Returns (ok, reason). `required` maps each function the
    code must define to its number of arguments."""
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
    for name, nargs in required.items():
        fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if not fn:
            return False, f"no function named {name}"
        if len(fn[0].args.args) != nargs:
            return False, f"{name} must take exactly {nargs} arguments"
    return True, "ok"


def compile_functions(src, required=RULE_FUNCS):
    """Validated source -> its namespace (functions and constants), executed
    with restricted builtins and only numpy available."""
    ok, why = validate(src, required)
    if not ok:
        raise ValueError(why)
    ns = {"np": np, "__builtins__": SAFE_BUILTINS}
    exec(compile(ast.parse(src), "<llm-code>", "exec"), ns)
    return ns


def run_in_child(target, args, timeout_s):
    """Run target(*args, q) in a spawned child and return the dict it puts on q,
    or a 'runtime' / 'timeout' failure. Polls rather than blocks, so a worker
    that dies without reporting (segfault, out-of-memory kill, failed spawn)
    fails in about a second instead of after timeout_s."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(*args, q))
    p.start()
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
        p.terminate()
        p.join(5)
        return {"ok": False, "stage": "timeout", "reason": f"no result within {timeout_s}s"}
    p.join(10)
    if p.is_alive():
        p.terminate()
    return res
