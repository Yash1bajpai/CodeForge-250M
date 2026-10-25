#!/usr/bin/env python3
"""Static interface audit of MBPP-style completions. Parses only; never executes model code.
Buckets (mutually exclusive, first match wins):
 noparse         completion not valid Python (ast.parse fails)
 nodef           parses, no top-level def at all
 unknown_target  target name/arity not determinable from asserts (no call, multi-arity callsites, starred/keyword args)
 name_missing    no top-level def with the target name (nested/method defs do not count: not callable as a global)
 unknown_def     final top-level def of that name is decorated, async, or has *args/**kwargs (arity not decidable statically)
 invalid_def     duplicate parameter names (compile-time SyntaxError that ast.parse does not raise)
 arity_ok        required<=k<=positional (defaults honored; required kw-only => arity_wrong)
 arity_wrong
Redefinition: the LAST top-level def of the name wins (runtime semantics). Callsites: every call to the target
name in every assert must have the same positional count, else unknown_target."""
import ast, collections, json, re, sys

def target_of(tail_or_prompt):
    """Return (name, k) or None. Uses only the asserts after the LAST "Your code should pass these tests:" marker (robust to corrupted task text)."""
    seg = tail_or_prompt.rsplit('Your code should pass these tests:', 1)[-1]
    asserts = re.findall(r'^assert .*$', seg, re.M)
    if not asserts: return None
    per = []
    for a in asserts:
        try: t = ast.parse(a)
        except SyntaxError: return None
        calls = [n for n in ast.walk(t) if isinstance(n, ast.Call)]
        per.append(calls)
    first = [c for c in per[0] if isinstance(c.func, ast.Name)]
    if not first: return None
    name = first[0].func.id
    ks = set()
    for calls in per:
        for c in calls:
            if isinstance(c.func, ast.Name) and c.func.id == name:
                if c.keywords or any(isinstance(x, ast.Starred) for x in c.args): return None
                ks.add(len(c.args))
    if len(ks) != 1: return None
    return name, ks.pop()

def classify(completion, tgt):
    try: tree = ast.parse(completion)
    except (SyntaxError, ValueError): return 'noparse'
    tops = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not tops: return 'nodef'
    if tgt is None: return 'unknown_target'
    name, k = tgt
    same = [d for d in tops if d.name == name]
    if not same: return 'name_missing'
    d = same[-1]
    a = d.args
    if isinstance(d, ast.AsyncFunctionDef) or d.decorator_list or a.vararg or a.kwarg: return 'unknown_def'
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg: names.append(a.vararg.arg)
    if len(set(names)) != len(names): return 'invalid_def'
    pos = len(a.posonlyargs) + len(a.args); lo = pos - len(a.defaults)
    if any(df is None for df in a.kw_defaults): return 'arity_wrong'
    return 'arity_ok' if lo <= k <= pos else 'arity_wrong'

def load_jsonl(p): return [json.loads(l) for l in open(p)]
