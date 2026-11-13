"""Finite, AST-only contradictions for one exact immutable built-in result.

A certificate exhausts a proven SUPERSET of values satisfying required domain
assertions. No task function is called. Unknown syntax/effects or resource limits
produce unknown, never a rejected candidate. Tests must run under the harness's
isolated pytest configuration, and the task must explicitly preserve semantics.
"""
from __future__ import annotations

import ast
import itertools
import os
import re
from pathlib import Path

from result_constraints import (
    ResultConstraintReport, _contract_text, _environment, _integrity_contract, _load,
    _result_call, _test_file,
)

_TYPES = {"str": str, "bytes": bytes, "tuple": tuple, "int": int}
_FULL_CHARS = [(0, 0x10FFFF)]
_MAX_CANDIDATES = 512
_MAX_OPERATIONS = 100000
_CONFIGS = {"conftest.py", "pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"}


class _Unknown(Exception):
    pass


class _Limit(_Unknown):
    pass


class _Meter:
    def __init__(self, limit):
        self.limit = limit
        self.operations = 0

    def step(self, count=1):
        self.operations += count
        if self.operations > self.limit:
            raise _Limit("Finite proof operation limit exceeded")


def _unknown(detail, evidence=()):
    return ResultConstraintReport(False, None, ("incomplete_analysis",),
                                  tuple(evidence) + ({"detail": detail},))


def _contract(instruction):
    text = _contract_text(instruction)
    if text is None:
        return False
    if _integrity_contract(text):
        return True
    sentences = {" ".join(s.lower().split()) for s in re.split(r"[.!?\n]+", text)}
    return {
        "all conditions apply to the same returned object, using ordinary python built-in semantics",
        "modifying tests, built-ins, globals or the test framework is forbidden",
    }.issubset(sentences)


def _inventory(root):
    """Reject ambiguous module mappings and uninspected Python/config surfaces."""
    modules = set()
    entries = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            entries += 1
            path = Path(directory) / name
            if entries > 4096 or path.is_symlink() or not path.resolve().is_relative_to(root):
                raise _Unknown("Incomplete/symlink workspace inventory")
            if name.lower() in _CONFIGS:
                raise _Unknown("Pytest configuration/fixtures are outside the sealed subset")
            if path.suffix.lower() in {'.pyc', '.pyo', '.pyd', '.so'}:
                raise _Unknown("Uninspected compiled Python module")
            if name.lower() in {'sitecustomize.py', 'usercustomize.py'}:
                raise _Unknown("Python startup hook is outside the sealed subset")
        for name in dirs:
            if name.startswith('.') or name in {"venv", "node_modules"}:
                raise _Unknown("Hidden/external directory is outside the sealed subset")
        dirs[:] = [name for name in dirs if name != "__pycache__"]
        for name in files:
            if name.endswith('.py'):
                if name.startswith('.'):
                    raise _Unknown("Hidden Python module")
                path = Path(directory) / name
                module = '.'.join(path.relative_to(root).with_suffix('').parts)
                if module.endswith('.__init__'):
                    module = module[:-len('.__init__')]
                if module in modules:
                    raise _Unknown("Ambiguous Python module path collision")
                modules.add(module)


def _required(node):
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return [part for item in node.values for part in _required(item)]
    return [node]


def _integer(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _integer(node.operand)
        if value is not None:
            return -value if isinstance(node.op, ast.USub) else value
    return None


def _type_identity(node):
    """Only type-singleton identity is independent of candidate object identity."""
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    if not isinstance(node.ops[0], (ast.Is, ast.IsNot)):
        return None
    left, right = node.left, node.comparators[0]
    if isinstance(left, ast.Name) and left.id in _TYPES:
        left, right = right, left
    if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
            and left.func.id == 'type' and len(left.args) == 1 and not left.keywords
            and isinstance(left.args[0], ast.Name)
            and isinstance(right, ast.Name) and right.id in _TYPES):
        return left.args[0].id, right.id, isinstance(node.ops[0], ast.Is)
    return None


def _length_equality(node, variable):
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return None
    left, right = node.left, node.comparators[0]
    for measure, constant in ((left, right), (right, left)):
        if (isinstance(measure, ast.Call) and isinstance(measure.func, ast.Name)
                and measure.func.id == 'len' and len(measure.args) == 1 and not measure.keywords
                and isinstance(measure.args[0], ast.Name) and measure.args[0].id == variable):
            return _integer(constant)
    return None


def _normalize(intervals):
    result = []
    for lower, upper in sorted(intervals):
        lower, upper = max(0, lower), min(0x10FFFF, upper)
        if lower > upper:
            continue
        if result and lower <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(result[-1][1], upper))
        else:
            result.append((lower, upper))
    return result


def _intersect(first, second):
    return _normalize([(max(a, c), min(b, d)) for a, b in first for c, d in second])


def _character_intervals(node, variable):
    """Unknown predicates denote the full Unicode range, a safe superset."""
    if isinstance(node, ast.BoolOp):
        children = [_character_intervals(value, variable) for value in node.values]
        if isinstance(node.op, ast.Or):
            return _normalize([interval for child in children for interval in child])
        intervals = _FULL_CHARS
        for child in children:
            intervals = _intersect(intervals, child)
        return intervals
    if not isinstance(node, ast.Compare):
        return _FULL_CHARS
    intervals = _FULL_CHARS
    left = node.left
    reverse = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE, ast.Eq: ast.Eq}
    for operation, right in zip(node.ops, node.comparators):
        value, constant, op = left, right, type(operation)
        if isinstance(right, ast.Name) and right.id == variable:
            value, constant, op = right, left, reverse.get(op)
        if (isinstance(value, ast.Name) and value.id == variable
                and isinstance(constant, ast.Constant) and type(constant.value) is str
                and len(constant.value) == 1):
            point = ord(constant.value)
            bounds = {ast.Lt: (0, point - 1), ast.LtE: (0, point),
                      ast.Gt: (point + 1, 0x10FFFF), ast.GtE: (point, 0x10FFFF),
                      ast.Eq: (point, point)}
            if op in bounds:
                intervals = _intersect(intervals, [bounds[op]])
        left = right
    return intervals


def _set_equality(node, variable):
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return None
    left, right = node.left, node.comparators[0]
    for measured, literal in ((left, right), (right, left)):
        if not (isinstance(measured, ast.Call) and isinstance(measured.func, ast.Name)
                and measured.func.id == 'set' and len(measured.args) == 1 and not measured.keywords
                and isinstance(measured.args[0], ast.Name) and measured.args[0].id == variable):
            continue
        if isinstance(literal, ast.Set):
            if len(literal.elts) > 5:
                raise _Unknown("Tuple set literal exceeds finite domain limit")
            values = [_integer(item) for item in literal.elts]
            if any(value is None for value in values):
                raise _Unknown("Tuple domain requires exact integer literals, not bool/float values")
            return set(values)
        if (isinstance(literal, ast.Call) and isinstance(literal.func, ast.Name)
                and literal.func.id == 'set' and not literal.args and not literal.keywords):
            return set()
    return None


def _all_integers(node, variable):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'all' and len(node.args) == 1 and not node.keywords
            and isinstance(node.args[0], ast.GeneratorExp)):
        return False
    generator = node.args[0]
    if len(generator.generators) != 1:
        return False
    clause = generator.generators[0]
    if (clause.ifs or clause.is_async or not isinstance(clause.target, ast.Name)
            or clause.target.id == variable or not isinstance(clause.iter, ast.Name)
            or clause.iter.id != variable):
        return False
    return _type_identity(generator.elt) == (clause.target.id, 'int', True)


class _Expressions:
    """A deliberately small pure expression language, checked before enumeration."""
    def __init__(self, variable, kind, type_line):
        self.variable, self.kind, self.type_line = variable, kind, type_line
        self.trusted_attributes = set()
        self.nodes = 0

    def validate(self, node, names=None, depth=0):
        names = {self.variable} if names is None else names
        self.nodes += 1
        if self.nodes > 4096 or depth > 60:
            raise _Limit("Predicate syntax limit exceeded")
        child = lambda value, scope=names: self.validate(value, scope, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (str, bytes, int, bool, type(None)):
            if isinstance(node.value, (str, bytes)) and len(node.value) > 4096:
                raise _Limit("Literal size limit exceeded")
            return
        if isinstance(node, ast.Name) and node.id in names | set(_TYPES):
            return
        if isinstance(node, (ast.Tuple, ast.Set)):
            if len(node.elts) > 512:
                raise _Limit("Literal container limit exceeded")
            for item in node.elts:
                child(item)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            child(node.operand)
            return
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for value in node.values:
                child(value)
            return
        if isinstance(node, ast.Compare):
            allowed = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot)
            if any(not isinstance(op, allowed) for op in node.ops):
                raise _Unknown("Unsupported comparison")
            if any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops) and _type_identity(node) is None:
                raise _Unknown("Arbitrary object identity is not a value constraint")
            child(node.left)
            for item in node.comparators:
                child(item)
            return
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
            child(node.value)
            child(node.slice)
            return
        if isinstance(node, ast.Call) and not node.keywords and not any(isinstance(x, ast.Starred) for x in node.args):
            if isinstance(node.func, ast.Name):
                arities = {'type': (1,), 'len': (1,), 'set': (0, 1), 'range': (1, 2, 3), 'all': (1,), 'any': (1,)}
                if node.func.id not in arities or len(node.args) not in arities[node.func.id]:
                    raise _Unknown("Unsupported builtin call")
                if node.func.id in {'all', 'any'} and not isinstance(node.args[0], ast.GeneratorExp):
                    raise _Unknown("all/any requires one bounded generator")
                for argument in node.args:
                    child(argument)
                return
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == self.variable and not node.args
                    and node.lineno > self.type_line
                    and node.func.attr in ({'lower', 'upper'} if self.kind == 'str' else {'hex'} if self.kind == 'bytes' else set())):
                self.trusted_attributes.add(id(node.func))
                return
        if isinstance(node, ast.GeneratorExp) and len(node.generators) == 1:
            clause = node.generators[0]
            if (clause.ifs or clause.is_async or not isinstance(clause.target, ast.Name)
                    or clause.target.id in names | set(_TYPES) | {'type', 'len', 'set', 'all', 'any', 'range'}):
                raise _Unknown("Generator rebinding/filter/async is outside the pure subset")
            iterable = clause.iter
            if not ((isinstance(iterable, ast.Name) and iterable.id == self.variable and self.kind == 'tuple')
                    or (isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name) and iterable.func.id == 'range')):
                raise _Unknown("Generator requires range or captured exact integer tuple")
            child(iterable)
            child(node.elt, names | {clause.target.id})
            return
        raise _Unknown("Unsupported expression: " + type(node).__name__)


def _evaluate(node, values, meter):
    meter.step()
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id] if node.id in values else _TYPES[node.id]
    if isinstance(node, (ast.Tuple, ast.Set)):
        items = [_evaluate(item, values, meter) for item in node.elts]
        return tuple(items) if isinstance(node, ast.Tuple) else set(items)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, values, meter)
        if isinstance(node.op, ast.Not):
            return not value
        return -value if isinstance(node.op, ast.USub) else +value
    if isinstance(node, ast.BoolOp):
        for item in node.values:
            result = _evaluate(item, values, meter)
            if isinstance(node.op, ast.And) and not result or isinstance(node.op, ast.Or) and result:
                return result
        return result
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, values, meter)
        for operation, item in zip(node.ops, node.comparators):
            right = _evaluate(item, values, meter)
            meter.step()
            if isinstance(operation, ast.Eq): result = left == right
            elif isinstance(operation, ast.NotEq): result = left != right
            elif isinstance(operation, ast.Lt): result = left < right
            elif isinstance(operation, ast.LtE): result = left <= right
            elif isinstance(operation, ast.Gt): result = left > right
            elif isinstance(operation, ast.GtE): result = left >= right
            elif isinstance(operation, ast.In): result = left in right
            elif isinstance(operation, ast.NotIn): result = left not in right
            elif isinstance(operation, ast.Is): result = left is right
            else: result = left is not right
            if not result:
                return False
            left = right
        return True
    if isinstance(node, ast.Subscript):
        value = _evaluate(node.value, values, meter)
        index = _evaluate(node.slice, values, meter)
        if type(value) not in (tuple, str, bytes) or type(index) not in (int, bool):
            raise _Unknown("Unmodeled indexing receiver/index")
        return value[index]
    if isinstance(node, ast.GeneratorExp):
        clause = node.generators[0]
        iterable = _evaluate(clause.iter, values, meter)
        if type(iterable) not in (range, tuple):
            raise _Unknown("Unmodeled generator iterable")
        try:
            size = len(iterable)
        except OverflowError:
            raise _Limit("Generator is too large") from None
        if size > 512:
            raise _Limit("Generator iteration limit exceeded")
        def generate():
            for item in iterable:
                meter.step()
                yield _evaluate(node.elt, dict(values, **{clause.target.id: item}), meter)
        return generate()
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            receiver = values[node.func.value.id]
            if type(receiver) is str and node.func.attr == 'lower': return str.lower(receiver)
            if type(receiver) is str and node.func.attr == 'upper': return str.upper(receiver)
            if type(receiver) is bytes and node.func.attr == 'hex': return bytes.hex(receiver)
            raise _Unknown("Method receiver is not the proven exact builtin")
        arguments = [_evaluate(item, values, meter) for item in node.args]
        name = node.func.id
        if name == 'type': return type(arguments[0])
        if name == 'len': return len(arguments[0])
        if name == 'set':
            if arguments and (type(arguments[0]) not in (tuple, str, bytes, range) or len(arguments[0]) > 512):
                raise _Limit("Set conversion outside bounded sequence subset")
            return set(*arguments)
        if name == 'range':
            if any(type(value) not in (int, bool) for value in arguments):
                raise _Unknown("range requires exact integer arguments")
            result = range(*arguments)
            try:
                if len(result) > 512: raise _Limit("Range limit exceeded")
            except OverflowError:
                raise _Limit("Range limit exceeded") from None
            return result
        if name == 'all': return all(arguments[0])
        if name == 'any': return any(arguments[0])
    raise _Unknown("Unmodeled expression")


def _block(function):
    if (function.args.posonlyargs or function.args.args or function.args.kwonlyargs
            or function.args.vararg or function.args.kwarg or function.decorator_list):
        return None
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and type(body[0].value.value) is str:
        body = body[1:]
    if (len(body) < 2 or not isinstance(body[0], ast.Assign) or len(body[0].targets) != 1
            or not isinstance(body[0].targets[0], ast.Name) or not isinstance(body[0].value, ast.Call)
            or any(not isinstance(stmt, ast.Assert) or stmt.msg is not None for stmt in body[1:])):
        return None
    variable = body[0].targets[0].id
    assertions = body[1:]
    guards = [(stmt.lineno, guard) for stmt in assertions for leaf in _required(stmt.test)
              if (guard := _type_identity(leaf)) is not None and guard[0] == variable and guard[2]]
    kinds = {guard[1] for _, guard in guards}
    if len(kinds) != 1 or next(iter(kinds)) not in {'str', 'bytes', 'tuple'}:
        return None
    kind = next(iter(kinds))
    expressions = _Expressions(variable, kind, min(line for line, _ in guards))
    for stmt in assertions:
        expressions.validate(stmt.test)
    return body[0], variable, kind, assertions, expressions.trusted_attributes


def _domain(variable, kind, assertions, maximum):
    leaves = [leaf for stmt in assertions for leaf in _required(stmt.test)]
    lengths = {_length_equality(leaf, variable) for leaf in leaves}
    lengths.discard(None)
    if kind == 'str':
        if 1 not in lengths:
            raise _Unknown("String domain requires an exact length-one assertion")
        intervals = _FULL_CHARS
        for stmt in assertions:
            intervals = _intersect(intervals, _character_intervals(stmt.test, variable))
        size = sum(upper - lower + 1 for lower, upper in intervals)
        if size > maximum:
            raise _Limit("String domain is not bounded within candidate limit")
        return [chr(point) for lower, upper in intervals for point in range(lower, upper + 1)], {'character_intervals': intervals}
    if kind == 'bytes':
        eligible = sorted(lengths & {0, 1})
        if not eligible:
            raise _Unknown("Bytes domain requires exact length zero or one")
        length = eligible[0]
        size = 1 if length == 0 else 256
        if size > maximum:
            raise _Limit("Bytes domain exceeds candidate limit")
        return ([b''] if length == 0 else [bytes([value]) for value in range(256)]), {'length': length}
    allowed = next((value for leaf in leaves if (value := _set_equality(leaf, variable)) is not None), None)
    if allowed is None or len(allowed) not in lengths or not any(_all_integers(leaf, variable) for leaf in leaves):
        raise _Unknown("Tuple domain requires exact length, exact integer elements and equal literal set")
    # Equal cardinality and exact set equality force each integer to occur once.
    size = 1
    for factor in range(2, len(allowed) + 1):
        size *= factor
    if len(allowed) > 5 or size > maximum:
        raise _Limit("Tuple permutation domain exceeds candidate limit")
    return list(itertools.permutations(sorted(allowed))), {'length': len(allowed), 'integer_set': sorted(allowed)}


def analyze_task(task, workspace, *, max_candidates=512, max_operations=100000,
                 max_files=80, max_total_bytes=1000000):
    if task.get('task_type') != 'coding':
        return ResultConstraintReport(False, None, (), ())
    if not _contract(task.get('instruction', '')):
        return _unknown("No explicit integrity and ordinary builtin semantics contract")
    if min(max_candidates, max_operations, max_files, max_total_bytes) <= 0:
        raise ValueError("Analysis limits must be positive")
    max_candidates = min(max_candidates, _MAX_CANDIDATES)
    meter = _Meter(min(max_operations, _MAX_OPERATIONS))
    evidence = []
    examined = 0
    try:
        path = Path(workspace)
        if path.is_symlink() or not path.is_dir():
            raise _Unknown("Missing/symlink workspace")
        root = path.resolve()
        _inventory(root)
        trees, error = _load(root, max_files, max_total_bytes)
        if error or not trees:
            raise _Unknown(error or "No Python source")
        blocks, trusted = [], set()
        for module, (path, source, tree) in trees.items():
            if not _test_file(path): continue
            for function in tree.body:
                if isinstance(function, ast.FunctionDef) and function.name.startswith('test_'):
                    if (function.args.posonlyargs or function.args.args or function.args.kwonlyargs
                            or function.args.vararg or function.args.kwarg):
                        raise _Unknown("Test fixture arguments are outside the sealed subset")
                    block = _block(function)
                    if block is not None:
                        blocks.append((module, path, source, function, block))
                        trusted.update(block[-1])
        namespaces, error = _environment(trees, extra_builtins=('bytes',),
                                         trusted_method_attributes=frozenset(trusted))
        if error:
            raise _Unknown(error)
        if not blocks:
            raise _Unknown("No supported linear captured-result test")
        for module, path, source, function, block in blocks:
            binding, variable, kind, assertions, _ = block
            if not _result_call(binding.value, namespaces[module]):
                raise _Unknown("Captured call does not resolve to an inspected local function")
            candidates, facts = _domain(variable, kind, assertions, max_candidates)
            proof = {'path': path.relative_to(root).as_posix(), 'test': function.name,
                     'binding_line': binding.lineno, 'variable': variable, 'builtin_type': kind,
                     'domain_size': len(candidates), 'exhausted_count': 0, 'domain_facts': facts,
                     'assert_lines': [stmt.lineno for stmt in assertions],
                     'assumptions': 'Explicit integrity contract; ordinary Python builtins; isolated pytest without external plugins/configuration'}
            satisfying = False
            for candidate in candidates:
                if examined >= max_candidates:
                    raise _Limit("Task-wide finite candidate limit exceeded")
                examined += 1
                meter.step()
                valid = True
                for stmt in assertions:
                    try:
                        if not _evaluate(stmt.test, {variable: candidate}, meter):
                            valid = False
                            break
                    except (IndexError, TypeError, ValueError, ZeroDivisionError):
                        # These are ordinary failures of fully supported pure expressions.
                        valid = False
                        break
                proof['exhausted_count'] += 1
                if valid:
                    satisfying = True
                    break
            proof['operations'] = meter.operations
            proof['examined_total'] = examined
            evidence.append(proof)
            if not satisfying:
                proof['proof'] = 'Exhausted a sound finite superset; no value satisfies every required assertion'
                return ResultConstraintReport(True, 'No exact built-in result satisfies the finite contract', (), (proof,))
        return ResultConstraintReport(False, None, (), tuple(evidence))
    except (_Unknown, OSError, UnicodeError, SyntaxError, ValueError, RecursionError) as exc:
        return _unknown(str(exc) or type(exc).__name__, evidence)