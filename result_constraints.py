"""Conservative contradictions on one captured built-in result, without executing inputs.

This intentionally supports only plain, linear, zero-argument pytest functions in
a fully inspected local Python workspace. Unknown imports, callbacks, fixtures,
shadowed builtins, mutable updates, and control flow disable hard conclusions.
It neither merges separate function calls nor requires a stateless task contract.
Only immutable int values and tuple lengths produce hard certificates; list length
may change asynchronously in a future implementation and is not a hard proof.
Hard results additionally require an explicit task contract forbidding changes to
tests, builtins, standard libraries, import hooks, and the test framework.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ResultConstraintReport:
    hard_impossible: bool
    reason: str | None
    flags: tuple[str, ...]
    evidence: tuple[dict, ...]

@dataclass
class _Domain:
    variable: str
    binding_line: int
    kind: str | None = None
    lower: int | None = None
    upper: int | None = None
    excluded: set[int] = field(default_factory=set)
    facts: list[dict] = field(default_factory=list)

_PURE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "frozenset", "int", "isinstance", "issubclass", "len", "list",
    "map", "max", "min", "pow", "range", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "ValueError", "TypeError", "NotImplementedError", "AssertionError",
    "OverflowError", "ZeroDivisionError",
}
# Higher-order invocation through unknown callbacks is intentionally unsupported.
_PURE_BUILTINS -= {"filter", "map", "issubclass"}
_PROTECTED = _PURE_BUILTINS | {"__builtins__", "object"}
_FORBIDDEN = (
    ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.AsyncFunctionDef,
    ast.Await, ast.Yield, ast.YieldFrom, ast.ClassDef, ast.Lambda,
)
_IGNORED = {".git", ".venv", "venv", "__pycache__", "node_modules"}

def _unknown(reason):
    return ResultConstraintReport(False, None, ("incomplete_analysis",), ({"detail": reason},))

def _literal(node):
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False

def _integer(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _integer(node.operand)
        if value is not None:
            return -value if isinstance(node.op, ast.USub) else value
    return None

def _plain_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        values = [_plain_names(e) for e in target.elts]
        return None if any(v is None for v in values) else [n for v in values for n in v]
    return None

def _test_file(path):
    return path.name.startswith("test_") or path.name.endswith("_test.py")

def _load(workspace, max_files, max_bytes):
    trees, total, entries = {}, 0, 0
    for directory, dirs, files in os.walk(workspace, followlinks=False):
        dirs[:] = [n for n in dirs if n not in _IGNORED and not n.startswith(".")]
        for name in dirs + files:
            entries += 1
            path = Path(directory) / name
            if entries > 4096 or path.is_symlink():
                return None, "Incomplete/symlink workspace inventory"
        for name in sorted(files):
            path = Path(directory) / name
            if name in {"conftest.py", "pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"}:
                return None, "Pytest fixtures/configuration may change runtime bindings"
            if path.suffix != ".py":
                continue
            size = path.stat().st_size
            if len(trees) >= max_files or size > 128_000 or total + size > max_bytes:
                return None, "Python source budget exceeded"
            total += size
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=str(path))
            module = ".".join(path.relative_to(workspace).with_suffix("").parts)
            if module.endswith(".__init__"):
                module = module[:-len(".__init__")]
            trees[module] = (path, source, tree)
    return trees, None

def _environment(trees, *, extra_builtins=(), trusted_method_attributes=frozenset()):
    """Validate a closed subset; optional method nodes must be proven pure by caller."""
    pure_builtins = _PURE_BUILTINS | set(extra_builtins)
    protected = _PROTECTED | set(extra_builtins)
    exports = {
        (module, n.name) for module, (_, _, tree) in trees.items()
        for n in tree.body if isinstance(n, ast.FunctionDef)
    }
    namespaces = {}
    for module, (path, _, tree) in trees.items():
        namespace = {name: ("function", module, name)
                     for mod, name in exports if mod == module}
        top_bound = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                new_names = [a.asname or a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                new_names = [] if node.module == "__future__" else [a.asname or a.name for a in node.names]
            elif isinstance(node, ast.FunctionDef):
                new_names = [node.name]
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                new_names = [name for target in targets for name in (_plain_names(target) or [])]
            else:
                new_names = []
            if len(new_names) != len(set(new_names)) or any(name in top_bound for name in new_names):
                return None, "Duplicate/overridden top-level binding"
            top_bound.update(new_names)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in trees or _test_file(trees[alias.name][0]):
                        return None, "Unknown/framework/test-module import"
                    namespace[alias.asname or alias.name] = ("module", alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module == "__future__" and all(a.name == "annotations" for a in node.names):
                    continue
                if node.level or node.module not in trees or _test_file(trees[node.module][0]):
                    return None, "Unknown/relative/framework/test-module import"
                for alias in node.names:
                    if (node.module, alias.name) not in exports:
                        return None, "Imported binding is not a local function"
                    namespace[alias.asname or alias.name] = ("function", node.module, alias.name)
            elif isinstance(node, ast.FunctionDef):
                if node.decorator_list or node.name in protected:
                    return None, "Decorator or builtin shadowing"
                if any(not _literal(v) for v in node.args.defaults + [
                        v for v in node.args.kw_defaults if v is not None]):
                    return None, "Computed function default"
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if value is None or not _literal(value):
                    return None, "Nonliteral module initialization"
                for target in targets:
                    names = _plain_names(target)
                    if names is None or any(n in protected or n in namespace for n in names):
                        return None, "Global/builtin/import rebinding"
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                pass
            else:
                return None, "Unsupported module initialization"
        if any(name in protected for name in namespace):
            return None, "Imported builtin shadowing"
        namespaces[module] = namespace

    for module, (_, _, tree) in trees.items():
        ns = namespaces[module]
        for node in ast.walk(tree):
            if isinstance(node, _FORBIDDEN):
                return None, "Unknown state, class, async, context manager, or callback"
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                return None, "Dynamic/dunder environment access"
            if isinstance(node, ast.arg) and node.arg in protected | set(ns):
                return None, "Argument shadows a builtin/import/function"
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                if node.id in protected | set(ns) | {"pytest_plugins", "pytestmark"}:
                    return None, "Local/global builtin or import rebinding"
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(_plain_names(t) is None for t in targets):
                    return None, "Mutable/attribute assignment"
            if isinstance(node, ast.Delete):
                return None, "Deletion may change aliases or bindings"
            if isinstance(node, ast.FunctionDef):
                if any(isinstance(child, (ast.FunctionDef, ast.Import, ast.ImportFrom))
                       for stmt in node.body for child in ast.walk(stmt)):
                    return None, "Nested function/import"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in pure_builtins and ns.get(node.func.id, (None,))[0] != "function":
                        return None, "Unknown callback/call"
                    if node.func.id == "sorted" and any(k.arg == "key" for k in node.keywords):
                        return None, "Unknown sorting callback"
                elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                      and ns.get(node.func.value.id, (None,))[0] == "module"):
                    target = ns[node.func.value.id][1]
                    if (target, node.func.attr) not in exports:
                        return None, "Unknown module attribute call"
                elif (isinstance(node.func, ast.Attribute)
                      and id(node.func) in trusted_method_attributes):
                    pass
                else:
                    return None, "Unknown method/call effects"
            elif isinstance(node, ast.Attribute):
                # Attributes are allowed only as the callee of an inspected local module.
                if not (isinstance(node.value, ast.Name)
                        and ns.get(node.value.id, (None,))[0] == "module"
                        and (ns[node.value.id][1], node.attr) in exports
                        or id(node) in trusted_method_attributes):
                    return None, "Unknown attribute access"
    return namespaces, None

def _result_call(node, namespace):
    if not isinstance(node, ast.Call):
        return False
    if any(isinstance(a, ast.Starred) or not _literal(a) for a in node.args):
        return False
    if any(k.arg is None or not _literal(k.value) for k in node.keywords):
        return False
    if isinstance(node.func, ast.Name):
        return namespace.get(node.func.id, (None,))[0] == "function"
    return (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
            and namespace.get(node.func.value.id, (None,))[0] == "module")

def _guard(node):
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Is) and len(node.comparators) == 1
            and isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Name)
            and node.left.func.id == "type" and len(node.left.args) == 1
            and not node.left.keywords and isinstance(node.left.args[0], ast.Name)
            and isinstance(node.comparators[0], ast.Name)
            and node.comparators[0].id in {"int", "tuple"}):
        return None
    return node.left.args[0].id, node.comparators[0].id

def _measure(node):
    if isinstance(node, ast.Name):
        return node.id, "int"
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "len" and len(node.args) == 1
            and not node.keywords and isinstance(node.args[0], ast.Name)):
        return node.args[0].id, "length"
    return None

def _constraint(node):
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and len(node.comparators) == 1):
        return None
    measure, value, op = _measure(node.left), _integer(node.comparators[0]), node.ops[0]
    if measure is None or value is None:
        measure, value = _measure(node.comparators[0]), _integer(node.left)
        reverse = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE,
                   ast.Eq: ast.Eq, ast.NotEq: ast.NotEq}
        if type(op) not in reverse:
            return None
        op = reverse[type(op)]()
    if measure is None or value is None or not isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
        return None
    return (*measure, type(op).__name__, value)

def _add(domain, op, value):
    lower, upper = None, None
    if op == "Eq":
        lower = upper = value
    elif op == "NotEq":
        domain.excluded.add(value)
    elif op == "Gt":
        lower = value + 1
    elif op == "GtE":
        lower = value
    elif op == "Lt":
        upper = value - 1
    elif op == "LtE":
        upper = value
    if lower is not None:
        domain.lower = lower if domain.lower is None else max(domain.lower, lower)
    if upper is not None:
        domain.upper = upper if domain.upper is None else min(domain.upper, upper)
    if domain.lower is None or domain.upper is None:
        return False
    if domain.lower > domain.upper:
        return True
    excluded = sum(domain.lower <= value <= domain.upper for value in domain.excluded)
    return excluded == domain.upper - domain.lower + 1

def _analyze_function(function, namespace, path, source):
    if (function.args.posonlyargs or function.args.args or function.args.kwonlyargs
            or function.args.vararg or function.args.kwarg or function.decorator_list):
        return None
    if any(isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.Match, ast.Return,
                          ast.Break, ast.Continue)) for n in ast.walk(function)):
        return None
    states = {}
    for stmt in function.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and _result_call(stmt.value, namespace)):
            # A call can change mutable aliases; never carry another result across it.
            states = {stmt.targets[0].id: _Domain(stmt.targets[0].id, stmt.lineno)}
            continue
        if not isinstance(stmt, ast.Assert) or stmt.msg is not None:
            states.clear()
            continue
        guard = _guard(stmt.test)
        if guard is not None:
            name, kind = guard
            state = states.get(name)
            if state is None or state.kind not in (None, kind):
                states.clear()
                continue
            state.kind = kind
            if kind in {"tuple"} and state.lower is None:
                state.lower = 0
            state.facts.append({"line": stmt.lineno, "assert": ast.get_source_segment(source, stmt)})
            continue
        constraint = _constraint(stmt.test)
        if constraint is None:
            states.clear()
            continue
        name, measure, op, value = constraint
        state = states.get(name)
        if state is None or state.kind is None:
            states.clear()
            continue
        if (measure == "int" and state.kind != "int") or (
                measure == "length" and state.kind not in {"tuple"}):
            states.clear()
            continue
        state.facts.append({"line": stmt.lineno, "assert": ast.get_source_segment(source, stmt)})
        if _add(state, op, value):
            return {
                "path": path, "test": function.name, "binding_line": state.binding_line,
                "variable": name, "builtin_type": state.kind, "measure": measure,
                "lower_inclusive": state.lower, "upper_inclusive": state.upper,
                "excluded": sorted(state.excluded), "facts": state.facts,
                "proof": "Empty integer domain for one captured built-in result; no intervening effects",
                "assumptions": "Task contract preserves tests, builtins, standard libraries, import hooks and framework",
            }
    return None

def _contract_text(instruction):
    """Remove quoted data; ambiguous quoting or normative exceptions are unknown."""
    text = str(instruction)
    if (re.search(r"(?m)^\s*>", text) or "~~~" in text
            or re.search(r"(?m)^(?: {4}|\t)", text)
            or re.search(r"<\s*(?:!--|/?[A-Za-z])", text)):
        return None
    if any(len(run) not in (1, 3) for run in re.findall(r"\x60+", text)):
        return None
    text = re.sub(r"(?s)\x60\x60\x60.*?\x60\x60\x60", " [quoted content] ", text)
    text = re.sub(r"(?s)“.*?”|‘.*?’", " [quoted content] ", text)
    text = re.sub(r"(?s)\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\x60[^\x60]*\x60", " [quoted content] ", text)
    if any(mark in text for mark in ('\"', "'", "“", "”", "‘", "’", chr(96))):
        return None
    if any(unicodedata.category(char) in {'Pi', 'Pf'} or char in '„‚「」『』〝〞〟＂＇' for char in text):
        return None
    if re.search(r"\b(?:except|unless|override|overrides|overridden|ignore|ignores|ignored|optional)\b", text, re.I):
        return None
    return text


def _integrity_contract(instruction):
    # Exact standalone normative sentence, after excluding quoted examples.
    text = _contract_text(instruction)
    if text is None:
        return False
    expected = ("do not modify tests, builtins, standard libraries, import hooks "
                "or the test framework")
    return any(" ".join(sentence.lower().split()) == expected
               for sentence in re.split(r"[.!?\n]+", text))


def analyze_task(task: dict[str, Any], workspace: Path, *, max_files=80,
                 max_total_bytes=1_000_000) -> ResultConstraintReport:
    if task.get("task_type") != "coding":
        return ResultConstraintReport(False, None, (), ())
    if not _integrity_contract(task.get("instruction", "")):
        return _unknown("No explicit contract preserving tests and builtin/framework semantics")
    if min(max_files, max_total_bytes) <= 0:
        raise ValueError("Source limits must be positive")
    root = Path(workspace).resolve()
    if not root.is_dir():
        return _unknown("Missing workspace")
    try:
        trees, error = _load(root, max_files, max_total_bytes)
    except (OSError, UnicodeError, SyntaxError, ValueError, RecursionError) as exc:
        return _unknown(type(exc).__name__)
    if error or not trees:
        return _unknown(error or "No Python sources")
    namespaces, error = _environment(trees)
    if error:
        return _unknown(error)
    for module, (path, source, tree) in trees.items():
        if not _test_file(path):
            continue
        for function in tree.body:
            if isinstance(function, ast.FunctionDef) and function.name.startswith("test_"):
                proof = _analyze_function(function, namespaces[module],
                                          path.relative_to(root).as_posix(), source)
                if proof:
                    return ResultConstraintReport(
                        True, "One captured built-in result has incompatible required constraints",
                        (), (proof,))
    return ResultConstraintReport(False, None, (), ())
