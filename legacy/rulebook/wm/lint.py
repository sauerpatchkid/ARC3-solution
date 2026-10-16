"""lint.py - the literal lint (docs/plans/rulebook-v2.md section 4.2): does a rule
have level facts written into its code?

An AST pass over the rule's functions (everything but bind(), whose job is to
guess level facts) that flags:

  colour literal      a literal colour 0-15 compared with board values, or written
                      into a board (use a PARAMS slot or api.background instead)
  coordinate literal  a literal above 3 used in an index or a slice
  coordinate pair     a literal pair of numbers such as (30, 4)

63 and 64 are the board's edge on every level and are never flagged, and neither
is anything compared with act.action. A rule with warnings is LEVEL-BOUND: it
can still be trusted on its own level, but it is left out of T1 until it is
rewritten, and the warnings go into the refinement prompt.

This is a heuristic: it reads names, not values. It can miss a level fact (a
literal row inside a comparison, a constant hidden in arithmetic) and can flag a
harmless one. The checker remains the judge of whether a rule is right.
"""
import ast

SMALL = 3                    # literals up to this size are offsets and steps, not coordinates
EDGE = (63, 64)              # the board's last index and its size
BOARDS = {"board", "board0", "grid", "out", "b", "before", "after", "frame", "new", "nb", "first"}
COLOUR_ATTRS = {"colour", "background", "first"}             # r.colour, api.background, api.first
KEEP_METHODS = {"copy", "astype", "ravel", "flatten", "reshape", "transpose", "max", "min", "item",
                "tolist", "squeeze"}                           # board.copy() still holds colours
KEEP_NP = {"where", "roll", "unique", "asarray", "array", "copy", "rot90", "flip", "fliplr", "flipud",
           "pad", "take", "transpose", "concatenate", "choose"}
KEEP_BUILTINS = {"int", "max", "min", "sorted", "list", "set", "tuple"}
ADVICE = {"colour literal": "use a PARAMS Colour() slot, or api.background",
          "coordinate literal": "use a PARAMS slot (Int, Offset or Region), or find the position on the board",
          "coordinate pair": "use a PARAMS slot (Offset or Region), or find the position on the board"}


def _int(node):
    """The value of an int literal (with an optional minus sign), else None."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _int(node.operand)
        return None if v is None else -v
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _colour_literal(node):
    """A literal colour, or a literal collection of them (for `in`)."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return bool(node.elts) and all(_colour_literal(e) for e in node.elts)
    v = _int(node)
    return v is not None and 0 <= v <= 15


def _holds_colours(node, names):
    """Does this expression hold board colours (as far as names can tell)?"""
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Subscript):
        return _holds_colours(node.value, names)
    if isinstance(node, ast.Attribute):
        return node.attr in COLOUR_ATTRS or (node.attr in ("T", "flat") and _holds_colours(node.value, names))
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in KEEP_METHODS:
            return _holds_colours(f.value, names)
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "np" and f.attr in KEEP_NP:
            return any(_holds_colours(a, names) for a in node.args)
        if isinstance(f, ast.Name) and f.id in KEEP_BUILTINS:
            return any(_holds_colours(a, names) for a in node.args)
    return False


def _board_names(fn, first_is_board):
    """Names in a function that hold board colours: its board argument, then
    anything assigned or looped from an expression that does."""
    args = [a.arg for a in fn.args.args]
    names = {a for a in args if a in BOARDS} | ({args[0]} if first_is_board and args else set())
    grew = True
    while grew:
        grew = False
        for n in ast.walk(fn):
            pairs = []
            if isinstance(n, ast.Assign):
                pairs = [(t, n.value) for t in n.targets]
            elif isinstance(n, (ast.AnnAssign, ast.NamedExpr)) and n.value is not None:
                pairs = [(n.target, n.value)]
            elif isinstance(n, (ast.For, ast.comprehension)):
                pairs = [(n.target, n.iter)]
            for target, value in pairs:
                if isinstance(target, ast.Name) and target.id not in names and _holds_colours(value, names):
                    names.add(target.id)
                    grew = True
    return names


def lint(src):
    """The rule's warnings, in line order: [{"kind", "line", "code", "advice"}].
    Empty for source that does not parse (the sandbox reports that)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    found = {}

    def flag(kind, node):
        code = (ast.get_source_segment(src, node) or "").strip().replace("\n", " ")
        found.setdefault((node.lineno, node.col_offset, kind),
                         {"kind": kind, "line": node.lineno, "code": code[:80], "advice": ADVICE[kind]})

    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name == "bind":
            continue
        names = _board_names(fn, first_is_board=fn.name in ("applies", "predict"))
        skip = set()                                           # literals that are not level facts
        for n in ast.walk(fn):
            if isinstance(n, ast.Compare) and any(isinstance(x, ast.Attribute) and x.attr == "action"
                                                  for x in [n.left, *n.comparators]):
                skip.update(id(x) for x in ast.walk(n))        # act.action == 4, act.action in (4, 5)
        for n in ast.walk(fn):
            # -- colour literals ---------------------------------------------------------
            if isinstance(n, ast.Compare):
                sides = [n.left, *n.comparators]
                for a, b in zip(sides, sides[1:]):
                    for holds, literal in ((a, b), (b, a)):
                        if _holds_colours(holds, names) and _colour_literal(literal):
                            flag("colour literal", n)
                            skip.update(id(x) for x in ast.walk(literal))      # (8, 9) here is colours, not a place
            elif isinstance(n, (ast.Assign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if _colour_literal(n.value) and not isinstance(n.value, (ast.Tuple, ast.List, ast.Set)) and any(
                        isinstance(t, ast.Subscript) and _holds_colours(t.value, names) for t in targets):
                    flag("colour literal", n)
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("where", "isin")
                  and isinstance(n.func.value, ast.Name) and n.func.value.id == "np"):
                rest = n.args[1:]
                if any(_colour_literal(a) for a in rest) and any(_holds_colours(a, names) for a in n.args):
                    flag("colour literal", n)
            # -- coordinate pairs ----------------------------------------------------------
            if isinstance(n, (ast.Tuple, ast.List)) and len(n.elts) == 2 and id(n) not in skip:
                vals = [_int(e) for e in n.elts]
                if None not in vals and max(abs(v) for v in vals) > SMALL and not all(v in EDGE for v in vals):
                    flag("coordinate pair", n)
                    skip.update(id(x) for x in ast.walk(n))
        for n in ast.walk(fn):
            # -- coordinate literals in an index or a slice ----------------------------------
            if isinstance(n, ast.Subscript):
                for x in ast.walk(n.slice):
                    v = _int(x)
                    if v is not None and abs(v) > SMALL and v not in EDGE and id(x) not in skip:
                        skip.add(id(x))
                        if isinstance(x, ast.UnaryOp):
                            skip.add(id(x.operand))
                        flag("coordinate literal", x)
    return [found[k] for k in sorted(found)]


def feedback(warnings):
    """The warnings as the refinement prompt shows them."""
    if not warnings:
        return ""
    lines = ["This rule has level facts written into its code, so it cannot carry to another level. "
             "Move each of these into PARAMS:"]
    lines += [f"  line {w['line']}: `{w['code']}` - {w['kind']}: {w['advice']}" for w in warnings]
    return "\n".join(lines)
