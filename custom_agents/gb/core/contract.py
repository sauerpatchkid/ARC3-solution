"""contract.py - the Rulebook v2 rule contract (docs/2-llm-history/rulebook-v2.md section 4.1).

v1's rules failed to carry to the next level because level facts (this colour,
that row) were written into the code. A v2 rule keeps its MECHANISM in code and
names every level-specific constant in PARAMS:

    RULE = "ACTION1 moves the agent block up one step if the cells above it are background"
    PARAMS = {"agent": Colour(), "step": Int(1, 8)}
    def bind(board0, api):                 # optional: a guess at PARAMS from the level's first board
        return {"agent": ...}
    def applies(board, act, api, p):       # p["agent"], p["step"]: this level's values
        ...
    def predict(board, act, api, p):       # -> next_board, or (next_board, claimed)
        ...

SLOT TYPES, and what the rule finds in `p`:
    Colour()         a colour index: one of the colours on the level's first board
                     or seen in a changed cell
    Int(lo, hi)      an integer, lo..hi inclusive
    Offset(max_abs)  a (drow, dcol) pair, each from -max_abs to max_abs
    Region()         one region of api.layout (the region itself, not its number)
    Choice([...])    one of the listed values

A BINDING is one value per slot, kept as plain data so it can be stored and sent
to a child process: an int, a [drow, dcol] pair, a region's NUMBER in the layout
of the evidence it was bound on, or a Choice's value. A region number means
nothing on another level, and not even on the other split of the same level
(the fit and score splits have different masks, so different layouts): whoever
runs a binding says which evidence's layout its regions are numbered in.

PARAMS is read STATICALLY: it must be a literal dict of slot calls with literal
arguments. The caller can then list a level's bindings without running any
LLM-written code; only bind(), applies() and predict() run, in a sandbox child.

A v1 rule (applies / predict with three arguments, no PARAMS) reads as
PARAMS = {} with the one empty binding, and claims every cell.

Glassbox (16 Oct): a rule compiled here may also call H1's names
(gb/h1_vocab.py SANDBOX: today `features`, which S3's fitted guards call),
imported on first use so that gb/core still loads without the Glassbox stages.
A rule that does not name them runs exactly as before.
"""
import ast
import itertools
import types

import numpy as np

from .evidence import GRID
from .evidence import Region as LayoutRegion
from .sandbox import compile_functions, validate

MAX_SLOTS = 6
MAX_BINDINGS = 4096          # more combinations than this and bind() must fix some slots (section 4.3)
MAX_INT_SPAN, MAX_OFFSET, MAX_CHOICES = 64, 8, 32


class RuleError(ValueError):
    """The rule's source does not follow the contract; str(e) is shown to the LLM."""


# --------------------------------------------------------------------------------
# Slot types
# --------------------------------------------------------------------------------
class Slot:
    def values(self, ev):
        """This slot's possible values on the level whose evidence is `ev`."""
        raise NotImplementedError

    def clean(self, value, api):
        """bind()'s answer for this slot as a binding value, or None if it is unusable."""
        raise NotImplementedError

    def resolve(self, value, layout):
        """What the rule is handed for a binding value."""
        return value


def _int(v):
    return int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else None


class Colour(Slot):
    def __init__(self):
        pass

    def values(self, ev):
        return level_colours(ev)

    def clean(self, value, api):
        v = _int(value)
        return v if v is not None and 0 <= v <= 15 else None


class Int(Slot):
    def __init__(self, lo, hi):
        if _int(lo) is None or _int(hi) is None or not lo <= hi or hi - lo >= MAX_INT_SPAN:
            raise RuleError(f"Int(lo, hi) needs integers with lo <= hi and at most {MAX_INT_SPAN} values")
        self.lo, self.hi = int(lo), int(hi)

    def values(self, ev):
        return list(range(self.lo, self.hi + 1))

    def clean(self, value, api):
        v = _int(value)
        return v if v is not None and self.lo <= v <= self.hi else None


class Offset(Slot):
    def __init__(self, max_abs):
        if _int(max_abs) is None or not 1 <= max_abs <= MAX_OFFSET:
            raise RuleError(f"Offset(max_abs) needs an integer from 1 to {MAX_OFFSET}")
        self.max_abs = int(max_abs)

    def values(self, ev):
        r = range(-self.max_abs, self.max_abs + 1)
        return [(dy, dx) for dy in r for dx in r]

    def clean(self, value, api):
        try:
            dy, dx = (_int(v) for v in value)
        except (TypeError, ValueError):
            return None
        return (dy, dx) if dy is not None and dx is not None and max(abs(dy), abs(dx)) <= self.max_abs else None


class Region(Slot):
    def __init__(self):
        pass

    def values(self, ev):
        return list(range(len(ev.api.layout)))

    def clean(self, value, api):
        if isinstance(value, LayoutRegion):                    # bind() answers with a region of api.layout
            return next((i for i, r in enumerate(api.layout) if r is value), None)
        v = _int(value)
        return v if v is not None and 0 <= v < len(api.layout) else None

    def resolve(self, value, layout):
        return layout[value]


class Choice(Slot):
    def __init__(self, options):
        ok = (isinstance(options, (list, tuple)) and 1 <= len(options) <= MAX_CHOICES
              and all(_plain(o) for o in options))
        if not ok:
            raise RuleError(f"Choice([...]) needs a list of 1 to {MAX_CHOICES} plain values "
                            "(numbers, strings, or tuples of them)")
        self.options = [plain(o) for o in options]

    def values(self, ev):
        return list(self.options)

    def clean(self, value, api):
        try:
            v = plain(value)
        except TypeError:
            return None
        return v if any(type(v) is type(o) and v == o for o in self.options) else None


def _plain(o):
    return (isinstance(o, (int, float, str, bool)) or o is None
            or (isinstance(o, (list, tuple)) and all(_plain(x) for x in o)))


def plain(o):
    """A binding value as plain, hashable data: lists as tuples (JSON gives
    lists back), numpy numbers as Python's."""
    if isinstance(o, (list, tuple)):
        return tuple(plain(x) for x in o)
    if isinstance(o, np.generic):
        return o.item()
    if not _plain(o):
        raise TypeError(type(o).__name__)
    return o


SLOT_TYPES = {"Colour": Colour, "Int": Int, "Offset": Offset, "Region": Region, "Choice": Choice}


def vocab():
    """The names a rule may call beside numpy and the slot types: H1's
    (gb/h1_vocab.py), imported here on first use."""
    from .. import h1_vocab
    return h1_vocab.SANDBOX
FUNCS = {1: {"applies": 3, "predict": 3}, 2: {"applies": 4, "predict": 4}}


def level_colours(ev):
    """A Colour slot's values on a level: the colours on its first board plus
    the colours seen in its changed (non-decoration) cells."""
    ch = (ev.before != ev.after) & ev.live
    seen = set(np.unique(ev.first).tolist()) | set(np.unique(ev.before[ch]).tolist()) \
        | set(np.unique(ev.after[ch]).tolist())
    return sorted(int(c) for c in seen)


# --------------------------------------------------------------------------------
# Reading a rule's form without running it
# --------------------------------------------------------------------------------
def read_params(src):
    """(version, params) for a rule's source, statically: version 1 or 2, and
    {slot name: Slot} in the order PARAMS lists them. Raises RuleError with a
    reason the LLM can act on."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise RuleError(f"syntax error: {e.msg} (line {e.lineno})")
    fns = {n.name: len(n.args.args) for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("applies", "predict"):
        if name not in fns:
            raise RuleError(f"no function named {name}")
    assigned = [n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                and any(isinstance(t, ast.Name) and t.id == "PARAMS"
                        for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    arity = (fns["applies"], fns["predict"])
    if arity == (3, 3):
        if assigned:
            raise RuleError("PARAMS is defined, so applies and predict must take a fourth argument: "
                            "applies(board, act, api, p) and predict(board, act, api, p)")
        return 1, {}
    if arity != (4, 4):
        raise RuleError("applies and predict must both take (board, act, api, p)")
    if "bind" in fns and fns["bind"] != 2:
        raise RuleError("bind must take exactly 2 arguments: bind(board0, api)")
    if not assigned:
        return 2, {}
    if len(assigned) > 1 or not isinstance(assigned[0].value, ast.Dict):
        raise RuleError("PARAMS must be written once, as a literal dict: PARAMS = {\"name\": Colour(), ...}")
    params = {}
    for k, v in zip(assigned[0].value.keys, assigned[0].value.values):
        if not (isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value.isidentifier()):
            raise RuleError("every PARAMS key must be a plain name in quotes")
        if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in SLOT_TYPES):
            raise RuleError(f"PARAMS[{k.value!r}] must be one of Colour(), Int(lo, hi), Offset(max_abs), "
                            "Region(), Choice([...])")
        try:
            args = [ast.literal_eval(a) for a in v.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in v.keywords}
            params[k.value] = SLOT_TYPES[v.func.id](*args, **kwargs)
        except RuleError:
            raise
        except (ValueError, TypeError, SyntaxError):
            raise RuleError(f"PARAMS[{k.value!r}]: {v.func.id}(...) needs plain literal arguments")
    if len(params) != len(assigned[0].value.keys):
        raise RuleError("PARAMS names a slot twice")
    if len(params) > MAX_SLOTS:
        raise RuleError(f"PARAMS has {len(params)} slots; at most {MAX_SLOTS} are allowed")
    return 2, params


def check_form(src):
    """Static check of a v1 or v2 rule: (ok, reason, version, params). The
    sandbox's rules (no imports, no private attributes, ...) plus the contract."""
    try:
        version, params = read_params(src)
    except RuleError as e:
        return False, str(e), None, None
    ok, why = validate(src, FUNCS[version])
    return ok, why, version, params


# --------------------------------------------------------------------------------
# Bindings (plain data; no LLM code runs here)
# --------------------------------------------------------------------------------
def domains(params, ev):
    """{slot name: its values on this level}, in PARAMS order."""
    return {name: slot.values(ev) for name, slot in params.items()}


def n_bindings(dom):
    n = 1
    for v in dom.values():
        n *= len(v)
    return n


def all_bindings(dom, fixed=None):
    """Every binding of the slots in `dom`, as dicts, in a fixed order; slots in
    `fixed` take that value only."""
    names = list(dom)
    pools = [[fixed[n]] if fixed and n in fixed else dom[n] for n in names]
    for combo in itertools.product(*pools):
        yield dict(zip(names, combo))


def freeze(binding):
    """A binding as a hashable, sortable key (and what JSON gives back, repaired)."""
    return tuple((k, plain(v)) for k, v in binding.items())


def describe(binding, params, layout=None):
    """A binding in words, for prompts and reports."""
    parts = []
    for name, v in binding.items():
        if isinstance(params.get(name), Region) and layout is not None and 0 <= v < len(layout):
            r = layout[v]
            v = f"region {v} (colour {r.colour}, {r.size} cells, rows {r.y0}-{r.y1}, cols {r.x0}-{r.x1})"
        parts.append(f"{name} = {v}")
    return ", ".join(parts) or "(no parameters)"


# --------------------------------------------------------------------------------
# Running a rule: ONLY inside a sandbox child process
# --------------------------------------------------------------------------------
class Rule:
    """A rule compiled in this process, v1 or v2 behind one way of calling it."""

    def __init__(self, src):
        ok, why, self.version, self.params = check_form(src)
        if not ok:
            raise ValueError(why)
        ns = compile_functions(src, FUNCS[self.version], extra={**SLOT_TYPES, **vocab()})
        self.text = str(ns.get("RULE", ""))[:300]
        self._applies, self._predict = ns["applies"], ns["predict"]
        self._bind = ns.get("bind") if self.version == 2 and self.params else None

    def values(self, binding, layout):
        """The read-only `p` a rule is handed for a binding. `layout`: the
        layout its region numbers refer to."""
        return types.MappingProxyType({name: slot.resolve(plain(binding[name]), layout)
                                       for name, slot in self.params.items()})

    def applies(self, board, act, api, p):
        return bool(self._applies(board, act, api, p) if self.version == 2 else self._applies(board, act, api))

    def predict(self, board, act, api, p, live):
        """(next board, claimed cells): predict()'s answer, checked. A bare
        board claims every cell; decoration cells are never claimed."""
        out = self._predict(board, act, api, p) if self.version == 2 else self._predict(board, act, api)
        claim = None
        if isinstance(out, tuple):
            if len(out) != 2:
                raise ValueError("predict must return a board or (board, claimed)")
            out, claim = out
        pred = np.asarray(out)
        if pred.shape != (GRID, GRID):
            raise ValueError(f"predict must return a 64x64 board, got shape {pred.shape}")
        if claim is None:
            return pred, live
        claim = np.asarray(claim)
        if claim.shape != (GRID, GRID):
            raise ValueError(f"claimed must be a 64x64 array, got shape {claim.shape}")
        return pred, claim.astype(bool) & live

    def guess(self, api):
        """bind()'s guess for this level as a (possibly partial) binding, and
        why it gave none: ({slot: value}, reason or None)."""
        if self._bind is None:
            return {}, None
        try:
            got = self._bind(api.first, api)
        except Exception as e:
            return {}, f"bind() failed: {type(e).__name__}: {e}"
        if not isinstance(got, dict):
            return {}, "bind() must return a dict of PARAMS values"
        out = {}
        for name, slot in self.params.items():
            if name in got:
                v = slot.clean(got[name], api)
                if v is not None:
                    out[name] = v
        return out, None
