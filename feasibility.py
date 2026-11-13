"""Conservative, bounded AST-only feasibility signals; never executes task code.

Hard impossibility follows incompatible integer-call contracts or an exhausted
finite domain for one captured exact immutable built-in result. Certificates
require explicit task semantics and bounded inspection of the test environment.
Incomplete analysis produces advisory signals for the planner.
"""
from __future__ import annotations
import ast
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class FeasibilityReport:
    hard_impossible: bool
    reason: str | None
    flags: tuple[str, ...]
    evidence: tuple[dict, ...]
    retry_hint: str

@dataclass(frozen=True)
class Ref:
    name: str

@dataclass(frozen=True)
class Value:
    tag: tuple

@dataclass(frozen=True)
class Call:
    function: str
    arguments: tuple
    keywords: tuple

_UNKNOWN = object()
_ALLOWED_FRAMEWORKS = {"pytest", "unittest", "math", "itertools", "collections",
    "functools", "heapq", "bisect", "operator", "re", "string", "typing",
    "statistics", "fractions"}
_IGNORED = {".git", ".venv", "venv", "__pycache__", "node_modules"}

def _test_path(path):
    return (path.name.startswith("test_") or path.name.endswith("_test.py")
            or any(part in {"test", "tests"} for part in path.parts))

def _literal(node, env, depth=0):
    if depth > 30:
        return _UNKNOWN
    if isinstance(node, ast.Name):
        result = env.get(node.id, _UNKNOWN)
        return result if isinstance(result, Value) else _UNKNOWN
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None:
            return Value(("none",))
        if type(value) is bool:
            return Value(("bool", value))
        if type(value) is int:
            return Value(("int", value))
        if type(value) is float and math.isfinite(value):
            return Value(("float", value.hex()))
        if type(value) in (str, bytes):
            return Value((type(value).__name__, value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand, env, depth + 1)
        if isinstance(value, Value) and value.tag[0] in {"int", "float"}:
            number = value.tag[1] if value.tag[0] == "int" else float.fromhex(value.tag[1])
            number = -number if isinstance(node.op, ast.USub) else number
            return Value(("int", number) if type(number) is int else ("float", number.hex()))
    if isinstance(node, ast.Tuple):
        values = [_literal(item, env, depth + 1) for item in node.elts]
        if all(isinstance(value, Value) for value in values):
            return Value(("tuple", tuple(value.tag for value in values)))
    return _UNKNOWN

def _number(value):
    if not isinstance(value, Value):
        return None
    kind = value.tag[0]
    if kind in {"int", "bool"}:
        return int(value.tag[1])
    if kind == "float":
        number = float.fromhex(value.tag[1])
        if number.is_integer():
            return int(number)
        return number
    return None

def _contract(instruction):
    from result_constraints import _contract_text
    plain = _contract_text(instruction) or ""
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", plain) if s.strip()]
    deterministic, integers = set(), set()
    stateful = bool(re.search(
        r"\b(?:stateful|call\s+count|maintain\s+(?:internal\s+)?state|"
        r"depends?\s+on\s+(?:previous|earlier)\s+calls|different\s+results\s+on\s+repeated)\b",
        plain, re.I))
    qualified = bool(re.search(r"\b(?:except|unless|optional|not\s+(?:stateless|deterministic))\b", plain, re.I))
    for sentence in sentences:
        found = re.fullmatch(
            r"(?:All )?functions must be (?:stateless and deterministic|deterministic and stateless)",
            sentence, re.I)
        if found:
            deterministic.add("*")
        found = re.fullmatch(
            r"(?:All )?functions must return (?:a )?built[- ]in (?:int|integer|integers)",
            sentence, re.I)
        if found:
            integers.add("*")
        found = re.fullmatch(
            r"(?:The )?function ([A-Za-z_]\w*) must be "
            r"(?:stateless and deterministic|deterministic and stateless)", sentence, re.I)
        if found:
            deterministic.add(found.group(1))
        found = re.fullmatch(
            r"(?:The )?function ([A-Za-z_]\w*) must return (?:a )?built[- ]in (?:int|integer)",
            sentence, re.I)
        if found:
            integers.add(found.group(1))
        found = re.fullmatch(
            r"Implement a (?:stateless deterministic|deterministic stateless|stateless and deterministic)"
            r" function ([A-Za-z_]\w*)\([^)]*\) returning a built[- ]in (?:int|integer)",
            sentence, re.I)
        if found:
            deterministic.add(found.group(1))
            integers.add(found.group(1))
    return deterministic, integers, stateful, qualified

class _Analysis:
    def __init__(self, trees, instruction):
        self.trees = trees
        self.instruction = instruction
        self.flags = set()
        self.facts = {}
        self.evidence = []
        self.exports = {}
        self.functions = set()
        self.conflicts = []
        self.max_facts = 2000
        self.fact_count = 0
        self.deterministic, self.integers, stateful, qualified = _contract(instruction)
        if stateful:
            self.flags.add("explicit_stateful")
        if qualified:
            self.flags.add("incomplete_analysis")
        # Dynamic import-time operations can replace a callable between two
        # otherwise identical aliases; never turn those observations into proof.
        known_roots = {module.split(".")[0] for module in trees}
        for module, (path, tree) in trees.items():
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    roots = ([node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module
                             else [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [])
                    if not getattr(node, "level", 0) and any(
                        root not in known_roots | _ALLOWED_FRAMEWORKS for root in roots):
                        self.flags.add("incomplete_analysis")
                if _test_path(Path(path)):
                    continue
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.decorator_list or any(isinstance(n, ast.Call)
                            for default in node.args.defaults + [v for v in node.args.kw_defaults if v is not None]
                            for n in ast.walk(default)):
                        self.flags.add("incomplete_analysis")
                elif isinstance(node, ast.Assign):
                    if (any(not isinstance(target, ast.Name) for target in node.targets)
                            or any(isinstance(n, (ast.Call, ast.NamedExpr)) for n in ast.walk(node.value))):
                        self.flags.add("incomplete_analysis")
                elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
                    pass
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    pass
                else:
                    self.flags.add("incomplete_analysis")
        # Definitions are indexed without evaluating a module or a function.
        for module, (_, tree) in trees.items():
            env = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    env[node.name] = Ref(module + "." + node.name + "@" + str(node.lineno))
                    self.functions.add(env[node.name].name)
                elif isinstance(node, ast.ClassDef):
                    env[node.name] = Ref(module + "." + node.name + "@" + str(node.lineno))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    self.imports(node, env, module, soft=False)
                elif isinstance(node, ast.Assign):
                    value = self.reference(node.value, env) or _literal(node.value, env)
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            env[target.id] = value
            self.exports[module] = env

    def imports(self, node, env, module, soft=True):
        if isinstance(node, ast.Import):
            for alias in node.names:
                env[alias.asname or alias.name.split(".")[0]] = Ref(
                    alias.name if alias.asname else alias.name.split(".")[0])
            return
        prefix = node.module or ""
        if node.level:
            is_package = self.trees.get(module, ("", None))[0].endswith("/__init__.py")
            parts = module.split(".") if is_package else module.split(".")[:-1]
            if node.level > 1:
                parts = parts[:-(node.level - 1)]
            prefix = ".".join(parts + ([prefix] if prefix else []))
        for alias in node.names:
            if alias.name == "*":
                if soft:
                    self.flags.add("incomplete_analysis")
                continue
            env[alias.asname or alias.name] = Ref(prefix + "." + alias.name)

    def reference(self, node, env):
        if isinstance(node, ast.Name):
            value = env.get(node.id)
            return value if isinstance(value, Ref) else None
        if isinstance(node, ast.Attribute):
            parent = self.reference(node.value, env)
            return Ref(parent.name + "." + node.attr) if parent else None
        return None

    def resolve(self, reference):
        if not reference:
            return None
        name, seen = reference.name, set()
        for _ in range(10):
            if name in seen:
                return None
            seen.add(name)
            if name in self.functions:
                return name
            module, _, symbol = name.rpartition(".")
            value = self.exports.get(module, {}).get(symbol)
            if not isinstance(value, Ref):
                return None
            name = value.name
        return None

    def expression(self, node, env):
        literal = _literal(node, env)
        if isinstance(literal, Value):
            return literal
        if isinstance(node, ast.Name):
            return env.get(node.id, _UNKNOWN)
        ref = self.reference(node, env)
        if ref:
            return ref
        if not isinstance(node, ast.Call):
            if isinstance(node, (ast.List, ast.Dict, ast.Set)):
                self.flags.add("mutable_arguments")
            return _UNKNOWN
        function = self.resolve(self.reference(node.func, env))
        if not function:
            self.flags.add("incomplete_analysis")
            return _UNKNOWN
        args, keywords = [], []
        for value in node.args:
            result = _literal(value, env)
            if not isinstance(result, Value):
                if isinstance(value, (ast.List, ast.Dict, ast.Set)) or (
                    isinstance(value, ast.Name) and env.get(value.id) == "mutable"):
                    self.flags.add("mutable_arguments")
                self.flags.add("incomplete_analysis")
                return _UNKNOWN
            args.append(result.tag)
        for keyword in node.keywords:
            if keyword.arg is None:
                self.flags.add("incomplete_analysis")
                return _UNKNOWN
            result = _literal(keyword.value, env)
            if not isinstance(result, Value):
                if isinstance(keyword.value, (ast.List, ast.Dict, ast.Set)) or (
                    isinstance(keyword.value, ast.Name) and env.get(keyword.value.id) == "mutable"):
                    self.flags.add("mutable_arguments")
                self.flags.add("incomplete_analysis")
                return _UNKNOWN
            keywords.append((keyword.arg, result.tag))
        # Preserve positional/keyword distinction AND kwargs order: a function
        # with **kwargs can observe ordering without violating determinism.
        origin = function.rsplit(".", 1)[0]
        if origin in self.trees and _test_path(Path(self.trees[origin][0])):
            self.flags.add("incomplete_analysis")
            return _UNKNOWN
        return Call(function, tuple(args), tuple(keywords))

    def equality(self, left, right, env, path, node):
        left, right = self.expression(left, env), self.expression(right, env)
        if isinstance(right, Call):
            left, right = right, left
        expected = _number(right)
        if not isinstance(left, Call) or expected is None:
            return
        self.fact_count += 1
        if self.fact_count > self.max_facts:
            self.flags.add("incomplete_analysis")
            return
        fact = {"path": path, "line": node.lineno, "function": left.function,
                "call": repr((left.arguments, left.keywords)), "expected": expected,
                "assertion": ast.unparse(node)[:600]}
        prior = self.facts.setdefault(left, [])
        for earlier in prior:
            if earlier["expected"] != expected:
                self.flags.add("conflicting_outputs")
                self.conflicts.append((left.function, earlier, fact))
                if len(self.evidence) < 20:
                    self.evidence.extend((earlier, fact))
                break
        prior.append(fact)

    def assertion(self, node, env, path):
        if isinstance(node, ast.Compare):
            values = [node.left] + node.comparators
            for index, operator in enumerate(node.ops):
                if isinstance(operator, ast.Eq):
                    self.equality(values[index], values[index + 1], env, path, node)
                else:
                    self.flags.add("incomplete_analysis")
        else:
            self.flags.add("incomplete_analysis")

    def block(self, body, env, path, module, unittest_self=None, module_scope=False):
        for statement in body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                self.imports(statement, env, module)
            elif isinstance(statement, ast.Assign):
                value = self.reference(statement.value, env)
                if value is None:
                    if isinstance(statement.value, (ast.List, ast.Dict, ast.Set)):
                        value = "mutable"
                    else:
                        value = self.expression(statement.value, env)
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = value
                    else:
                        self.flags.add("incomplete_analysis")
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                env[statement.target.id] = self.expression(statement.value, env) if statement.value else _UNKNOWN
            elif isinstance(statement, ast.Assert):
                self.assertion(statement.test, env, path)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                continue
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                is_self = (isinstance(call.func, ast.Attribute) and
                           isinstance(call.func.value, ast.Name) and call.func.value.id == unittest_self)
                if is_self and call.func.attr == "assertEqual" and len(call.args) == 2 and not call.keywords:
                    self.equality(call.args[0], call.args[1], env, path, statement)
                elif is_self and call.func.attr == "assertTrue" and len(call.args) == 1 and not call.keywords:
                    self.assertion(call.args[0], env, path)
                else:
                    self.flags.add("incomplete_analysis")
            elif isinstance(statement, ast.Return):
                break
            elif isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
                if module_scope:
                    env[statement.name] = Ref(module + "." + statement.name + "@" + str(statement.lineno))
                else:
                    self.flags.add("incomplete_analysis")
            elif isinstance(statement, ast.Pass):
                continue
            else:
                # Never merge facts from alternate branches, loops, exceptions,
                # context managers, monkeypatches or unknown execution.
                self.flags.add("incomplete_analysis")

    def variants(self, function, env):
        variants = [dict(env)]
        parameter_names = set()
        for decorator in function.decorator_list:
            ref = self.reference(decorator.func, env) if isinstance(decorator, ast.Call) else None
            if not ref or ref.name != "pytest.mark.parametrize" or len(decorator.args) != 2:
                self.flags.add("incomplete_analysis")
                return []
            if any(k.arg not in {"ids"} for k in decorator.keywords):
                self.flags.add("incomplete_analysis")
                return []
            for keyword in decorator.keywords:
                values = keyword.value.elts if isinstance(keyword.value, (ast.List, ast.Tuple)) else [keyword.value]
                if not all(isinstance(value, ast.Constant) and (value.value is None or isinstance(value.value, str))
                           for value in values):
                    self.flags.add("incomplete_analysis")
                    return []
            names_node, rows_node = decorator.args
            if not isinstance(names_node, ast.Constant) or not isinstance(names_node.value, str):
                self.flags.add("incomplete_analysis")
                return []
            names = [name.strip() for name in names_node.value.split(",")]
            if (not all(name.isidentifier() for name in names) or len(set(names)) != len(names)
                    or parameter_names.intersection(names) or not isinstance(rows_node, (ast.List, ast.Tuple))):
                self.flags.add("incomplete_analysis")
                return []
            parameter_names.update(names)
            expanded = []
            for variant in variants:
                for row in rows_node.elts:
                    values = [row] if len(names) == 1 else row.elts if isinstance(row, (ast.Tuple, ast.List)) else []
                    if len(values) != len(names):
                        self.flags.add("incomplete_analysis")
                        continue
                    mapping = dict(variant)
                    for name, value in zip(names, values):
                        mapping[name] = "mutable" if isinstance(value, (ast.List, ast.Dict, ast.Set)) else _literal(value, variant)
                    expanded.append(mapping)
                    if len(expanded) > 128:
                        self.flags.add("incomplete_analysis")
                        return []
            variants = expanded
        arguments = {argument.arg for argument in function.args.posonlyargs + function.args.args
                     + function.args.kwonlyargs}
        if arguments != parameter_names or function.args.vararg or function.args.kwarg:
            self.flags.add("incomplete_analysis")
            return []
        return variants

    def run(self):
        for module, (path, tree) in self.trees.items():
            if not _test_path(Path(path)) and Path(path).name != "conftest.py":
                continue
            if Path(path).name == "conftest.py":
                self.flags.add("incomplete_analysis")
                continue
            env = dict(self.exports.get(module, {}))
            # Top-level aliases/constants are evaluated abstractly, not executed.
            self.block(tree.body, env, path, module, module_scope=True)
            definitions = {node.name: node for node in tree.body
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            for name, node in definitions.items():
                if env.get(name) != Ref(module + "." + name + "@" + str(node.lineno)):
                    self.flags.add("incomplete_analysis")
                    continue
                if isinstance(node, ast.FunctionDef) and node.decorator_list and not name.startswith("test"):
                    self.flags.add("incomplete_analysis")
                if isinstance(node, ast.FunctionDef) and name.startswith("test"):
                    for variant in self.variants(node, env):
                        self.block(node.body, variant, path, module)
                elif isinstance(node, ast.ClassDef) and any(
                    self.reference(base, env) == Ref("unittest.TestCase") for base in node.bases):
                    if (len(node.bases) != 1 or node.decorator_list or node.keywords or any(
                        not isinstance(child, (ast.FunctionDef, ast.Pass))
                        and not (isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant))
                        for child in node.body)):
                        self.flags.add("incomplete_analysis")
                        continue
                    methods = {child.name: child for child in node.body if isinstance(child, ast.FunctionDef)}
                    if any(name in methods for name in ("setUp", "tearDown", "setUpClass", "tearDownClass",
                                                        "assertEqual", "assertTrue", "run")):
                        self.flags.add("incomplete_analysis")
                        continue
                    for method_name, method in methods.items():
                        if not method_name.startswith("test"):
                            continue
                        if (method.decorator_list or len(method.args.args) != 1 or
                                method.args.posonlyargs or method.args.kwonlyargs or
                                method.args.vararg or method.args.kwarg):
                            self.flags.add("incomplete_analysis")
                            continue
                        self.block(method.body, dict(env), path, module, method.args.args[0].arg)
        hard = False
        if "incomplete_analysis" not in self.flags and "explicit_stateful" not in self.flags:
            for function, _, _ in self.conflicts:
                name = function.rsplit(".", 1)[-1].split("@")[0]
                if (("*" in self.deterministic or name in self.deterministic)
                        and ("*" in self.integers or name in self.integers)):
                    hard = True
                    break
        reason = ("incompatible numeric outputs for identical immutable inputs under explicit "
                  "stateless deterministic built-in-int contract") if hard else None
        hint = "formal_skip" if hard else "avoid_expensive_retry" if self.flags else "normal"
        return FeasibilityReport(hard, reason, tuple(sorted(self.flags)), tuple(self.evidence), hint)

def analyze_task(task: dict[str, Any], workspace: Path, *, max_files: int = 80,
                 max_total_bytes: int = 1_000_000, max_file_bytes: int = 128_000) -> FeasibilityReport:
    """Analyze without importing/executing task files. Flags are advisory only."""
    if min(max_files, max_total_bytes, max_file_bytes) <= 0:
        raise ValueError("Analysis limits must be positive")
    root = Path(workspace).resolve()
    instruction = str(task.get("instruction", ""))
    flags = set()
    if not root.is_dir():
        return FeasibilityReport(False, None, ("missing_evidence",), (), "avoid_expensive_retry")
    for name in re.findall(r"(?<![\w/])([\w/-]+\.(?:py|csv|md|json|txt))\b", instruction):
        candidate = root / name
        if not candidate.exists() and not any(root.glob("*/" + Path(name).name)):
            flags.add("missing_evidence")
    if task.get("task_type") != "coding":
        return FeasibilityReport(False, None, tuple(sorted(flags)), (),
                                 "avoid_expensive_retry" if flags else "normal")
    trees, total, entries = {}, 0, 0
    # Walk without materializing an unbounded recursive list.
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries > 4096:
                        flags.add("incomplete_analysis")
                        pending.clear()
                        break
                    path = Path(entry.path)
                    if entry.is_symlink():
                        flags.add("incomplete_analysis")
                        continue
                    try:
                        path.resolve().relative_to(root)
                    except (ValueError, OSError):
                        flags.add("incomplete_analysis")
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in _IGNORED:
                            pending.append(path)
                        continue
                    if not entry.is_file(follow_symlinks=False) or path.suffix != ".py":
                        continue
                    if len(trees) >= max_files:
                        flags.add("incomplete_analysis")
                        continue
                    size = entry.stat(follow_symlinks=False).st_size
                    if size > max_file_bytes or total + size > max_total_bytes:
                        flags.add("incomplete_analysis")
                        continue
                    with path.open("rb") as stream:
                        raw = stream.read(min(max_file_bytes, max_total_bytes - total) + 1)
                    if len(raw) > max_file_bytes or total + len(raw) > max_total_bytes:
                        flags.add("incomplete_analysis")
                        continue
                    total += len(raw)
                    try:
                        tree = ast.parse(raw.decode("utf-8-sig"), filename=str(path))
                    except (UnicodeError, SyntaxError, ValueError, RecursionError):
                        flags.add("incomplete_analysis")
                        continue
                    relative = path.relative_to(root)
                    module = ".".join(relative.with_suffix("").parts)
                    if module.endswith(".__init__"):
                        module = module[:-9]
                    trees[module] = (relative.as_posix(), tree)
        except OSError:
            flags.add("incomplete_analysis")
    if not any(_test_path(Path(path)) for path, _ in trees.values()):
        flags.add("missing_evidence")
    analysis = _Analysis(trees, instruction)
    analysis.flags.update(flags)
    report = analysis.run()
    if report.hard_impossible or "incomplete_analysis" in flags:
        return report
    from finite_constraints import analyze_task as analyze_finite_contract
    try:
        finite = analyze_finite_contract(task, root, max_files=max_files,
                                         max_total_bytes=max_total_bytes)
    except Exception:
        return report
    if finite.hard_impossible:
        return FeasibilityReport(True, finite.reason, ("finite_domain_contradiction",),
                                 finite.evidence, "formal_skip")
    return report

