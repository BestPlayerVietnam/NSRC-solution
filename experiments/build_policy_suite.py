"""Build 144 new authored tasks; validate gold offline in a networkless Docker.

64 train + 32 calibration + 48 test. Structural families never cross splits.
Only task.json and workspace go to a solver. Metadata, oracle and references
are evaluator-only. This is a synthetic policy-development set, not a claim
about the hidden competition distribution.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import itertools
import json
import math
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "suite_policy"
SEED = 20261113091
SPLITS = ["train"] * 4 + ["calibration"] * 2 + ["test"] * 3


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source(text):
    return textwrap.dedent(text).strip() + "\n"


def csv_text(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue()


def scalar(value):
    if value is None:
        return None
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def safe(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or ":" in name:
        raise ValueError("unsafe path")
    return path


class Builder:
    def __init__(self):
        self.items = []

    def add(self, domain, family, variant, instruction, files, expected, **oracle):
        ident = "policy_" + digest(f"{SEED}:{domain}:{family}:{variant}")[:16]
        self.items.append(dict(id=ident, domain=domain, family=family, variant=variant,
            instruction=instruction, files=files, expected=expected, oracle=oracle))

    def variants(self, family):
        negative = int(digest(f"negative:{SEED}:{family}")[:8], 16) % 4
        for variant in range(4):
            yield variant, variant == negative, random.Random(int(digest(f"{SEED}:{family}:{variant}")[:16], 16))


MATH = [
    ("pack_yield", "(a*b-c)//d", "A cooperative receives {a} sacks containing {b} seeds each. It removes {c} damaged seeds and makes complete planting packets of {d} seeds. How many complete packets result?"),
    ("rental_billing", "a*b+c*d-e", "A studio rents equipment for {a} hours at {b} cents per hour and lighting for {c} hours at {d} cents per hour. A coupon removes {e} cents from the combined bill. What is the final bill in cents?"),
    ("recipe_scale", "(a*b)//c", "A recipe uses {b} grams of spice for {c} servings. A kitchen plans {a} servings and scales proportionally, rounding the total spice down to a whole number of grams. How many grams does it use?"),
    ("parallel_trips", "a+b*c-d*e", "A depot has {a} meters of cable. It receives {b} reels of {c} meters and completes {d} installations consuming {e} meters each. How many meters remain? Negative balances mean an amount owed."),
    ("border_and_spare", "2*(a+b)*c+d", "There are {c} rectangular signs, each {a} cm long and {b} cm wide. Every sign needs trim around its entire perimeter, and the workshop adds {d} cm of spare trim in total. How many centimeters of trim are required?"),
    ("discount_then_share", "((a*b*9)//10-c)//d", "A club receives {a} payments of {b} cents. A service keeps 10 percent of the combined payments; the club receives the remaining 90 percent rounded down to whole cents. It pays {c} cents of postage and divides the remainder into {d} equal whole-cent grants, keeping any leftover cents. What is one grant?"),
    ("ceiling_shipping", "(a*b-c+d-1)//d", "A publisher prints {a} bundles of {b} cards and sets aside {c} cards. A shipping case holds at most {d} cards. Partial cases are allowed. What is the minimum number of cases needed for all remaining cards?"),
    ("work_share", "(a*b)//(b+c)", "A contract allocates {a} bonus cents between two teams in proportion to their completed units: the first team completed {b} units and the second {c}. Round the first team's allocation down to whole cents. How many cents does the first team receive?"),
    ("tiered_charge", "min(a,b)*c+max(a-b,0)*d+e", "A service processes {a} parcels. The first {b} parcels cost {c} cents each, all later parcels cost {d} cents each, and one fixed handling charge is {e} cents. What is the total charge in cents?"),
]


def math_eval(expr, values):
    return eval(expr, {"__builtins__": {}, "min": min, "max": max}, values)


def math_cases(builder):
    for family, expr, template in MATH:
        for variant, negative, rng in builder.variants(family):
            scale = [1, 2, 6, 3][variant]
            values = dict(a=rng.randint(45, 80)*scale, b=rng.randint(8, 19), c=rng.randint(2, 7), d=rng.randint(2, 8), e=rng.randint(3, 14))
            displayed = dict(values)
            oracle = dict(validation_kind="math", expression=expr, variables=values)
            if negative:
                displayed["a"] = "an unrecorded number of"
                first = dict(values); second = dict(values, a=values["a"]+101)
                answers = [math_eval(expr, first), math_eval(expr, second)]
                assert answers[0] != answers[1]
                oracle.update(worlds=[first, second], witness_answers=answers,
                    reason="The unrecorded first quantity can take both witness values; all other facts are unchanged.")
                expected = None
            else:
                expected = scalar(math_eval(expr, values))
            question = template.format(**displayed)
            question += " Use only the stated facts. Return a single integer; if no unique answer follows, return UNSOLVABLE."
            builder.add("math", family, variant, question, {}, expected, **oracle)


# Each algorithm is confined to one split. Reference code is never a solver file.
CODE = [
    ("comparison_adapter", "choose", "Return an object that compares equal to either element of allowed, with ordinary equality semantics otherwise. A custom class is permitted; the return type is deliberately not constrained to int.",
     "def choose(allowed):\n    return allowed[0]\n",
     source('''
         class Choice:
             def __init__(self, allowed): self.allowed = tuple(allowed)
             def __eq__(self, other): return other in self.allowed
         def choose(allowed): return Choice(allowed)
     ''')),
    ("explicit_stateful_counter", "reset, next_value", "reset(start) sets an internal integer counter. Each next_value() returns the current counter and increments it by one. State is explicitly permitted and required between calls; reset must discard earlier state.",
     "_value = 0\ndef reset(start):\n    global _value\n    _value = start\ndef next_value():\n    return _value\n",
     "_value = 0\ndef reset(start):\n    global _value\n    _value = start\ndef next_value():\n    global _value\n    value = _value\n    _value += 1\n    return value\n"),
    ("unweighted_shortest_path", "distance", "distance(edges, start, target) returns the smallest number of directed edges from start to target, or -1 if unreachable. An empty path from a vertex to itself has length zero. Vertices are integers and cycles may occur.",
     "def distance(edges, start, target):\n    return 0 if start == target else -1\n",
     source('''
         from collections import deque
         def distance(edges, start, target):
             queue = deque([(start, 0)])
             seen = {start}
             adjacency = {}
             for a, b in edges: adjacency.setdefault(a, []).append(b)
             while queue:
                 node, depth = queue.popleft()
                 if node == target: return depth
                 for other in adjacency.get(node, []):
                     if other not in seen:
                         seen.add(other); queue.append((other, depth+1))
             return -1
     ''')),
    ("ragged_transpose", "transpose", "transpose(rows) transposes a possibly ragged list of lists. The output has as many rows as the longest input row; missing cells are filled with None. Return lists, preserve input row order, and do not mutate input. Empty input returns [].",
     "def transpose(rows):\n    return [list(column) for column in zip(*rows)]\n",
     "def transpose(rows):\n    width = max((len(row) for row in rows), default=0)\n    return [[row[i] if i < len(row) else None for row in rows] for i in range(width)]\n"),
    ("largest_remainder_allocation", "allocate", "allocate(total, weights) distributes a nonnegative integer total in proportion to positive integer weights. First give each entry floor(total*weight/sum(weights)); assign the remaining units in descending fractional-remainder order, breaking ties by smaller index. Return a list of built-in ints.",
     "def allocate(total, weights):\n    return [total*w//sum(weights) for w in weights]\n",
     source('''
         def allocate(total, weights):
             denominator = sum(weights)
             result = [total*w//denominator for w in weights]
             order = sorted(range(len(weights)), key=lambda i: (-(total*weights[i] % denominator), i))
             for i in order[:total-sum(result)]: result[i] += 1
             return result
     ''')),
    ("signed_comma_parser", "parse_numbers", "parse_numbers(text) reads comma-separated signed decimal integers, allowing surrounding whitespace on every token. Empty or whitespace-only input returns []. Empty internal tokens or malformed integer tokens raise ValueError. Return a list of built-in ints.",
     "def parse_numbers(text):\n    return [int(piece.strip().strip('+-')) for piece in text.split(',')] if text.strip() else []\n",
     source('''
         import re
         def parse_numbers(text):
             if not text.strip(): return []
             result = []
             for piece in text.split(','):
                 piece = piece.strip()
                 if not re.fullmatch(r'[+-]?[0-9]+', piece): raise ValueError('invalid integer')
                 result.append(int(piece))
             return result
     ''')),
    ("dag_longest_levels", "levels", "levels(vertices, edges) returns a dict mapping every vertex to the length of the longest directed path ending there. Source and isolated vertices have level 0. For a directed cycle raise ValueError. Do not modify inputs.",
     "def levels(vertices, edges):\n    return dict.fromkeys(vertices, 0)\n",
     source('''
         def levels(vertices, edges):
             parents = {v: [] for v in vertices}
             for a, b in edges: parents[b].append(a)
             result = {}; visiting = set()
             def depth(node):
                 if node in visiting: raise ValueError('cycle')
                 if node in result: return result[node]
                 visiting.add(node)
                 result[node] = max((depth(p)+1 for p in parents[node]), default=0)
                 visiting.remove(node)
                 return result[node]
             for node in vertices: depth(node)
             return result
     ''')),
    ("run_length_expansion", "expand", "expand(runs) expands a list of pairs (one-character string, nonnegative integer count) by repeating each character count times and concatenating in order. Return a built-in str. Zero counts contribute nothing.",
     "def expand(runs):\n    return ''.join(char for char, count in runs)\n",
     "def expand(runs):\n    return ''.join(char*count for char, count in runs)\n"),
    ("stable_three_way_partition", "partition", "partition(values, pivot) returns a tuple of three lists: values less than pivot, equal to pivot, and greater than pivot. Preserve original relative order within every list and do not change the input. All inputs and output elements are built-in ints.",
     "def partition(values, pivot):\n    return tuple(sorted(x for x in values if test(x)) for test in (lambda x:x<pivot, lambda x:x==pivot, lambda x:x>pivot))\n",
     "def partition(values, pivot):\n    return ([x for x in values if x<pivot], [x for x in values if x==pivot], [x for x in values if x>pivot])\n"),
]


def code_checks(family, k, public_pass):
    """Public examples and separately authored edge checks, no generated solver."""
    if family == "comparison_adapter":
        pub = f"value = choose(({k}, {k+2}))\nassert value == {k}\n" + ("" if public_pass else f"assert value == {k+2}\n")
        hidden = f"value = choose(({k+4}, {k+9}))\nassert value == {k+4}\nassert value == {k+9}\nassert not value == {k+3}\n"
        bad = f"value = choose(({k}, {k+2}))\nassert type(value) is int\nassert value == {k}\nassert value == {k+2}\n"
        rule = f"For the fixed input ({k}, {k+2}), choose must return a built-in int equal to both {k} and {k+2}. Statelessness and determinism are required."
        proof = "Distinct equalities on one built-in integer. The satisfiable sibling explicitly permits a custom equality object."
    elif family == "explicit_stateful_counter":
        pub = f"reset({k})\nassert next_value() == {k}\n" + ("" if public_pass else f"assert next_value() == {k+1}\n")
        hidden = f"reset({k+7})\nassert [next_value() for _ in range(4)] == {list(range(k+7,k+11))!r}\nreset(0)\nassert next_value() == 0\n"
        bad = f"reset({k})\na = next_value()\nb = next_value()\nassert type(a) is int and type(b) is int\nassert a == {k} and b == {k+1}\n"
        rule = f"After reset({k}), next_value is additionally required to be stateless and deterministic: every call must return the same integer, while its first two calls must return {k} and {k+1}."
        proof = "A deterministic stateless no-argument function cannot produce two different built-in ints in the same unchanged state."
    elif family == "unweighted_shortest_path":
        pub = "assert distance([], 0, 1) == -1\n" if public_pass else f"assert distance([(0, {k})], 0, {k}) == 1\n"
        hidden = f"assert distance([(0, 1), (1, 2), (2, 0), (2, {k})], 0, {k}) == 3\nassert distance([], {k}, {k}) == 0\nassert distance([(1, 2)], 2, 1) == -1\n"
        bad = f"value = distance([], {k}, {k})\nassert type(value) is int\nassert value == 0\nassert value > 0\n"
        rule = "For an empty path from a vertex to itself, the returned built-in integer must be zero and strictly positive simultaneously."
        proof = "Zero cannot be strictly positive for a built-in integer."
    elif family == "ragged_transpose":
        pub = "assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]\n" if public_pass else f"assert transpose([[{k}], [{k+1}, {k+2}]]) == [[{k}, {k+1}], [None, {k+2}]]\n"
        hidden = f"a = [[{k}, {k+1}], [], [{k+2}]]\nassert transpose(a) == [[{k}, None, {k+2}], [{k+1}, None, None]]\nassert a == [[{k}, {k+1}], [], [{k+2}]]\nassert transpose([]) == []\n"
        bad = "value = transpose([[1], [2, 3]])\nassert type(value) is list\nassert len(value) == 1\nassert len(value) == 2\n"
        rule = "The public tests impose two simultaneous row counts on the same returned built-in list. All tests are part of the contract."
        proof = "No built-in list has length both 1 and 2; no output-int contract is needed."
    elif family == "largest_remainder_allocation":
        pub = "assert allocate(6, [1, 2]) == [2, 4]\n" if public_pass else "assert allocate(5, [1, 1, 1]) == [2, 2, 1]\n"
        hidden = f"assert allocate({k}, [1]) == [{k}]\nassert allocate(1, [1, 1, 1]) == [1, 0, 0]\nassert allocate(7, [2, 1]) == [5, 2]\nassert allocate(0, [1, 3]) == [0, 0]\n"
        bad = f"value = allocate({k}, [1, 1])\nassert type(value) is list and len(value) == 2\nassert all(type(x) is int and x == 0 for x in value)\nassert sum(value) == {k}\n"
        rule = f"For total={k}, weights=[1,1], both allocated built-in integers must be zero while their sum must equal the positive total."
        proof = "Two zeros cannot sum to a positive total."
    elif family == "signed_comma_parser":
        pub = "assert parse_numbers('1, +2') == [1, 2]\n" if public_pass else f"assert parse_numbers(' -{k}, +2 ') == [-{k}, 2]\n"
        hidden = "assert parse_numbers('') == []\nassert parse_numbers(' +0, -12, 9 ') == [0, -12, 9]\ntry:\n    parse_numbers('1,,2')\nexcept ValueError:\n    pass\nelse:\n    assert False\n"
        bad = "value = parse_numbers('1')\nassert type(value) is list\nassert len(value) >= 2\nassert len(value) <= 1\n"
        rule = "On input '1', return a built-in list whose length is at least 2 and at most 1."
        proof = "The allowed length interval is empty."
    elif family == "dag_longest_levels":
        pub = "assert levels([1, 2], []) == {1: 0, 2: 0}\n" if public_pass else "assert levels([1, 2], [(1, 2)]) == {1: 0, 2: 1}\n"
        hidden = f"assert levels([0, 1, 2, {k}], [(0,1),(1,2),(0,2)]) == {{0:0,1:1,2:2,{k}:0}}\ntry:\n    levels([0,1], [(0,1),(1,0)])\nexcept ValueError:\n    pass\nelse:\n    assert False\n"
        bad = "value = levels([0,1], [(0,1),(1,0)])\nassert type(value) is dict\nassert type(value[0]) is int and type(value[1]) is int\nassert value[0] >= value[1]+1\nassert value[1] >= value[0]+1\n"
        rule = "For the two-vertex directed cycle, do not raise: instead return built-in integer levels satisfying level[0]>=level[1]+1 and level[1]>=level[0]+1."
        proof = "Adding the two required inequalities yields 0>=2."
    elif family == "run_length_expansion":
        pub = "assert expand([('a',1),('b',1)]) == 'ab'\n" if public_pass else "assert expand([('a',3),('b',0),('c',2)]) == 'aaacc'\n"
        hidden = f"assert expand([]) == ''\nassert expand([('Ж',2),('x',0),('q',3)]) == 'ЖЖqqq'\nassert expand([('z',{k})]) == 'z'*{k}\n"
        bad = "value = expand([('x',0)])\nassert type(value) is str\nassert len(value) == -1\n"
        rule = "For [('x',0)], return a built-in string of length -1."
        proof = "Built-in string lengths are nonnegative."
    else:
        pub = "assert partition([1,2,3], 2) == ([1],[2],[3])\n" if public_pass else "assert partition([8,1,7,0,4], 4) == ([1,0],[4],[8,7])\n"
        hidden = f"a = [{k+2},1,{k+1},0,{k}]\nassert partition(a,{k}) == ([1,0],[{k}],[{k+2},{k+1}])\nassert a == [{k+2},1,{k+1},0,{k}]\nassert partition([],0) == ([],[],[])\n"
        bad = f"value = partition([{k}], {k})\nx = value[0][0]\nassert type(x) is int\nassert x < {k}\nassert x > {k}\n"
        rule = f"The first element of the first returned list must be a built-in int simultaneously less than and greater than {k}."
        proof = "The strict lower and upper bounds exclude every integer."
    return pub, hidden, bad, rule, proof


def pytest_file(import_line, checks):
    return import_line + "\ndef test_contract():\n" + textwrap.indent(checks, "    ")


def code_cases(builder):
    for family, api, description, broken, reference in CODE:
        positive_index = 0
        for variant, negative, rng in builder.variants(family):
            k = rng.randint(21, 60)
            pass_public = not negative and positive_index == 1
            if not negative: positive_index += 1
            pub, hidden, bad, extra, proof = code_checks(family, k, pass_public)
            nested = variant == 2
            filename = "engine/logic.py" if nested else "mylib.py"
            module = "engine.logic" if nested else "mylib"
            files = {filename: broken}
            if nested: files["engine/__init__.py"] = ""
            public_path = "tests/unit/test_contract.py" if nested else "test_contract.py"
            files[public_path] = pytest_file(f"from {module} import {api}", bad if negative else pub)
            question = f"Repair {filename}. API: {api}. {extra if negative else description} "
            question += "Public pytest tests are part of the specification. Only implementation files may be changed. Do not modify tests, builtins, standard libraries, import hooks or the test framework. Return DONE after repair, or UNSOLVABLE if no implementation can satisfy the contract."
            builder.add("coding", family, variant, question, files, None if negative else "PASS_HIDDEN",
                validation_kind="coding", reason=proof if negative else None,
                reference_files={} if negative else {filename: reference},
                hidden_files={"test_evaluator_hidden.py": pytest_file(f"from {module} import {api}", bad if negative else hidden)},
                public_profile="public_pass_hidden_fail" if pass_public else "public_fail",
                proof_type=family if negative else None)


SEARCH = ["latest_record", "alias_resolution", "interval_roster", "labelled_route", "committee_intersection",
          "priority_override", "partial_order_predecessor", "conditional_exception", "longest_streak"]


def search_answer(kind, data):
    if kind == "latest_record":
        t = max(r["revision"] for r in data["records"])
        return sorted({r["status"] for r in data["records"] if r["revision"] == t})
    if kind == "alias_resolution":
        name = data["query"]
        for _ in range(8):
            if name in data["badges"]: return [data["badges"][name]]
            if name not in data["aliases"]: return []
            name = data["aliases"][name]
        return []
    if kind == "interval_roster":
        return sorted({r["person"] for r in data["roster"] if r["start"] <= data["minute"] < r["end"]})
    if kind == "labelled_route":
        possible = {data["start"]}
        for label in data["labels"]:
            possible = {r["to"] for r in data["edges"] if r["from"] in possible and r["label"] == label}
        return sorted(possible)
    if kind == "committee_intersection":
        return sorted(set.intersection(*(set(x) for x in data["committees"])))
    if kind == "priority_override":
        score = max((r["priority"], r["revision"]) for r in data["records"])
        return sorted({r["value"] for r in data["records"] if (r["priority"],r["revision"]) == score})
    if kind == "partial_order_predecessor":
        possible = set()
        for order in itertools.permutations(data["events"]):
            if all(order.index(a) < order.index(b) for a,b in data["before"]):
                index = order.index(data["target"])
                possible.add(order[index-1] if index else "NO_PREDECESSOR")
        return sorted(possible)
    if kind == "conditional_exception":
        if data["exception_active"] is None: return [data["default_owner"],data["exception_owner"]]
        return [data["exception_owner"] if data["exception_active"] else data["default_owner"]]
    streaks = []; start = None
    for row in data["rows"] + [{"minute": -1, "ok": False}]:
        if row["ok"]:
            if start is None: start = row["minute"]; count = 0
            count += 1
        elif start is not None:
            streaks.append((count,start)); start = None
    longest = max(n for n,t in streaks)
    return [str(t) for n,t in streaks if n == longest]


def search_cases(builder):
    for family in SEARCH:
        for variant, negative, rng in builder.variants(family):
            tag = digest(f"{family}:{variant}:{SEED}")[:6]
            a,b,c,d = [prefix+tag for prefix in ("Aster_","Birch_","Cedar_","Dune_")]
            if family == "latest_record":
                data = {"records":[{"revision":i,"status":s} for i,s in [(1,a),(3,b),(2,c)]]}
                if negative: data["records"].append({"revision":3,"status":d})
                question = "What status is uniquely determined by the highest revision record? Equal revisions have equal authority; there is no tie-breaking rule."
            elif family == "alias_resolution":
                data = {"query":a,"aliases":{a:b,b:c},"badges":{c:"BADGE-"+tag,d:"TOKEN-"+tag}}
                if negative: del data["aliases"][b]
                question = f"Follow the recorded aliases for {a}. What badge belongs to its final canonical identity? An unrecorded alias has no assumed target."
            elif family == "interval_roster":
                t = rng.randint(15,35)
                data = {"minute":t,"roster":[{"person":a,"start":0,"end":t},{"person":b,"start":t,"end":60}]}
                if negative: data["roster"].append({"person":c,"start":t-1,"end":t+2})
                question = f"Who is uniquely on duty at minute {t}? Each interval includes its start and excludes its end; there is no priority among roster entries."
            elif family == "labelled_route":
                data = {"start":a,"labels":["east","north"],"edges":[{"from":a,"label":"east","to":b},{"from":b,"label":"north","to":c}]}
                if negative: data["edges"].append({"from":b,"label":"north","to":d})
                question = f"Starting at {a}, follow east once and then north once. What destination is uniquely determined? No edge has priority over another with the same label."
            elif family == "committee_intersection":
                data = {"committees":[[a,b],[b,c],[b,d]]}
                if negative: data["committees"] = [[a,b,c],[a,b,d],[a,b]]
                question = "Which single person belongs to all three listed committees? If more than one fits, no unique person is determined."
            elif family == "priority_override":
                data = {"records":[{"priority":1,"revision":99,"value":a},{"priority":3,"revision":2,"value":b},{"priority":3,"revision":1,"value":c}]}
                if negative: data["records"].append({"priority":3,"revision":2,"value":d})
                question = "Which configuration value wins? Higher priority overrides lower priority; among equal priorities higher revision wins. Equal priority and revision have no further precedence rule."
            elif family == "partial_order_predecessor":
                data = {"events":[a,b,c,d],"target":d,"before":[[a,b],[b,c],[c,d]]}
                if negative: data["before"] = [[a,b],[a,c],[b,d],[c,d]]
                question = f"All events occur exactly once. The before relations are the only ordering constraints. Which event must be immediately before {d}?"
            elif family == "conditional_exception":
                data = {"default_owner":a,"exception_owner":b,"exception_active":None if negative else bool(variant%2)}
                question = "Which owner applies? Use exception_owner exactly when exception_active is true, otherwise default_owner. A null active flag means it was not recorded, not false."
            else:
                flags = [True,True,False,True,True,True,False,True]
                if negative: flags = [True,True,False,True,True,False]
                offset = rng.randint(10,50)
                data = {"rows":[{"minute":offset+i,"ok":flag} for i,flag in enumerate(flags)]}
                question = "At what minute does the unique longest consecutive run of ok=true begin? Rows are consecutive one-minute records. If multiple runs have the same maximum length, no unique starting minute is determined."
            answers = search_answer(family,data)
            assert (len(answers) != 1) == negative
            path = f"kb/records_{rng.randint(100,999)}.json"
            files = {path:dumps(data)}
            for i in range([2,8,30,15][variant]):
                files[f"kb/archive/{i:03}.txt"] = f"Inventory note {i}: shelf {1000+i} contains sealed spare equipment. This note gives no rule or fact about the requested records.\n"
            witnesses = []
            if family == "alias_resolution" and negative:
                for target in (c,d):
                    world = json.loads(json.dumps(data))
                    world["aliases"][b] = target
                    witnesses.append(world)
            builder.add("search",family,variant,question+" Read only the complete supplied knowledge base. Return the exact requested value, or UNSOLVABLE when the facts do not determine one unique value.", files,
                None if negative else answers[0], validation_kind="search", search_kind=family, data_file=path,
                candidates=answers, witness_worlds=witnesses, reason=("The missing alias can point to either of two recorded canonical identities with different badges." if not answers else "At least two values satisfy the stated rules; none has precedence.") if negative else None)


DATA = ["grouped_maximum", "weighted_average", "conditional_distinct", "anti_join_total", "dated_net", "joined_prices", "rolling_peak", "second_rank", "percentage_share"]


def sql_answer(files, tables, query):
    with sqlite3.connect(":memory:") as con:
        for name,spec in tables.items():
            rows = list(csv.reader(io.StringIO(files[spec["file"]])))
            quote = lambda x:'"'+x.replace('"','""')+'"'
            schema = ",".join(quote(c)+" "+t for c,t in zip(rows[0],spec["types"]))
            con.execute("CREATE TABLE "+quote(name)+" ("+schema+")")
            con.executemany("INSERT INTO "+quote(name)+" VALUES ("+",".join("?" for _ in rows[0])+")", [[None if x=="" else x for x in row] for row in rows[1:]])
        value = con.execute(query).fetchone()[0]
        return scalar(value)


def data_cases(builder):
    for family in DATA:
        for variant,negative,rng in builder.variants(family):
            n = [12,28,96,44][variant]
            rows = [[i+1,"g"+str(i%4),rng.randint(4,45),rng.randint(1,7),"ok" if i%3 else "hold"] for i in range(n)]
            files = {"observations.csv":csv_text(["id","category","amount","weight","status"],rows)}
            tables = {"observations":{"file":"observations.csv","types":["INTEGER","TEXT","INTEGER","INTEGER","TEXT"]}}
            if family == "grouped_maximum":
                query = "SELECT MAX(total) FROM (SELECT SUM(amount) total FROM observations WHERE status='ok' GROUP BY category)"
                question = "For each category sum amount on ok rows; return the largest of these category totals."
            elif family == "weighted_average":
                query = "SELECT SUM(amount*weight)/SUM(weight) FROM observations"
                question = "Return the weighted mean of amount using weight, rounded down to an integer."
            elif family == "conditional_distinct":
                for i,row in enumerate(rows): row[1] = "item"+str(i//2)
                files["observations.csv"] = csv_text(["id","category","amount","weight","status"],rows)
                query = "SELECT COUNT(DISTINCT category) FROM observations WHERE amount>=20 AND status='ok'"
                question = "How many distinct categories have at least one ok row whose amount is at least 20?"
            elif family == "anti_join_total":
                files["excluded.csv"] = csv_text(["id"],[[i+1] for i in range(0,n,4)])
                tables["excluded"] = {"file":"excluded.csv","types":["INTEGER"]}
                query = "SELECT SUM(amount) FROM observations o WHERE NOT EXISTS (SELECT 1 FROM excluded e WHERE e.id=o.id)"
                question = "Sum amount for observation IDs that do not appear in excluded.csv."
            elif family == "dated_net":
                rows = [[f"2025-03-{i%28+1:02}",rng.randint(3,50),"refund" if i%4==0 else "sale"] for i in range(n)]
                files = {"journal.csv":csv_text(["Дата","Сумма","Вид"],rows)}
                tables = {"journal":{"file":"journal.csv","types":["TEXT","INTEGER","TEXT"]}}
                query = "SELECT SUM(CASE WHEN \"Вид\"='sale' THEN \"Сумма\" ELSE -\"Сумма\" END) FROM journal WHERE \"Дата\" BETWEEN '2025-03-05' AND '2025-03-20'"
                question = "In journal.csv, return net Сумма for dates March 5 through March 20, 2025 inclusive. Add sale rows and subtract refund rows."
            elif family == "joined_prices":
                rows = [[i+1,"sku"+str(i%5),rng.randint(1,8)] for i in range(n)]
                files = {"receipts.csv":csv_text(["receipt id","sku","units"],rows),"prices.csv":csv_text(["sku","unit price"],[["sku"+str(i),10+3*i] for i in range(5)])}
                tables = {"receipts":{"file":"receipts.csv","types":["INTEGER","TEXT","INTEGER"]},"prices":{"file":"prices.csv","types":["TEXT","INTEGER"]}}
                query = 'SELECT SUM(r.units*p."unit price") FROM receipts r JOIN prices p ON r.sku=p.sku'
                question = "Join receipts.csv to prices.csv by sku. Return total units times unit price across all receipts; price keys are unique."
            elif family == "rolling_peak":
                query = "SELECT MAX(total) FROM (SELECT SUM(amount) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) total FROM observations)"
                question = "Sort observations by id. At every row sum amount from that row and up to two previous rows. Return the largest resulting running total, including shorter prefixes."
            elif family == "second_rank":
                query = "SELECT SUM(amount) FROM (SELECT amount, ROW_NUMBER() OVER (PARTITION BY category ORDER BY amount DESC,id) r FROM observations) WHERE r=2"
                question = "Within each category sort by amount descending, breaking ties by id ascending. Add the amount of the second row in every category. All categories contain at least two rows."
            else:
                query = "SELECT 100*SUM(CASE WHEN status='ok' THEN amount ELSE 0 END)/SUM(amount) FROM observations"
                question = "What percentage of total amount belongs to ok rows? Round the percentage down to a whole integer. All amounts are nonnegative and the total is positive."
            expected = sql_answer(files,tables,query)
            oracle = dict(validation_kind="sql",tables=tables,query=query)
            if negative:
                chosen = None
                # Find a genuinely influential measure cell; never alter IDs or keys.
                for table,spec in tables.items():
                    parsed = list(csv.reader(io.StringIO(files[spec["file"]])))
                    for column,name in enumerate(parsed[0]):
                        if name not in {"amount","Сумма","units","unit price"}: continue
                        for rownum in range(1,len(parsed)):
                            worlds = []
                            answers = []
                            for value in (1,100003):
                                replacement = [list(row) for row in parsed]
                                replacement[rownum][column] = str(value)
                                world = dict(files); world[spec["file"]] = csv_text(replacement[0],replacement[1:])
                                worlds.append(world); answers.append(sql_answer(world,tables,query))
                            if answers[0] is not None and answers[1] is not None and answers[0] != answers[1]:
                                chosen = (spec["file"],parsed,rownum,column,worlds,answers); break
                        if chosen: break
                    if chosen: break
                assert chosen, family
                filename,parsed,rownum,column,worlds,answers = chosen
                parsed[rownum][column] = ""
                files[filename] = csv_text(parsed[0],parsed[1:])
                oracle.update(world_files=worlds,witness_answers=answers,
                    reason="One unrecorded measure cell can take either nonnegative witness value, producing different answers with identical observed cells.")
                expected = None
            question += " Numeric cells are numeric, including quoted thresholds. A blank numeric cell means an unknown nonnegative value, not zero or a row to drop. Use only supplied files; return a single integer or UNSOLVABLE if the result is not uniquely determined."
            builder.add("data_analysis",family,variant,question,files,expected,**oracle)


def write_suite(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing suite; use --validate-only")
    builder = Builder()
    math_cases(builder); code_cases(builder); search_cases(builder); data_cases(builder)
    family_splits = {domain:{family:SPLITS[i] for i,family in enumerate(families)} for domain,families in (
        ("math",[x[0] for x in MATH]),("coding",[x[0] for x in CODE]),("search",SEARCH),("data_analysis",DATA))}
    assert len(builder.items) == 144
    metadata,oracle,records = {},{},[]
    reward_rng = random.Random(SEED ^ 0x93EF)
    for item in sorted(builder.items,key=lambda x:x["id"]):
        ident = item["id"]; folder = output/"tasks"/ident; work = folder/"workspace"
        work.mkdir(parents=True)
        task = dict(task_id=ident,task_type=item["domain"],reward=reward_rng.randint(1,5),
                    instruction=item["instruction"]+f"\nCase identifier: {ident}.")
        (folder/"task.json").write_text(dumps(task),encoding="utf-8")
        for filename,content in item["files"].items():
            path = work/safe(filename); path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8")
        records.append(dict(id=ident,domain=item["domain"],reward=task["reward"],task_path="tasks/"+ident,
            task_sha256=digest(dumps(task)),files_sha256={name:digest(body) for name,body in item["files"].items()}))
        metadata[ident] = dict(split=family_splits[item["domain"]][item["family"]],
            group=item["domain"]+":"+item["family"],source_family=item["family"],size_variant=item["variant"])
        oracle[ident] = {"expected":item["expected"],"answerable":item["expected"] is not None,
            "hidden_files":{},"reference_files":{},"reason":None,**item["oracle"]}
    manifest = dict(version=1,seed=SEED,label="synthetic-policy-development",task_count=144,
        description="64 train, 32 calibration, 48 test; disjoint structural families. Not representative hidden-final evidence.",
        oracle_sha256=digest(dumps(oracle)),research_metadata_sha256=digest(dumps(metadata)),tasks=records)
    for name,value in (("manifest.json",manifest),("research_metadata.json",metadata),("oracle.json",oracle)):
        (output/name).write_text(dumps(value),encoding="utf-8")
    (output/"README.md").write_text(source('''
        # Synthetic policy-development suite
        144 newly authored tasks: 64 train, 32 calibration, 48 heldout test.
        Each domain contributes 16/8/12 tasks. Every structural family contributes
        three answerable cases and one proved negative; negative variant positions
        are independently seeded rather than fixed by size. Groups never cross splits.
        Rewards are independently seeded integers 1..5, unrelated to difficulty.

        Give solvers only tasks/<id>/task.json and workspace/. Keep oracle.json,
        research_metadata.json, validation.json and this README outside solver input.
        The solver manifest has no answerability or family labels; split and group
        live in research_metadata.json. Hidden tests and reference code are oracle-only.
        Coding includes explicit state/custom-equality near-negatives, nine distinct
        contradiction mechanisms, and public-pass/hidden-fail originals. A public
        failure indicates a repair opportunity, not a guarantee that a model retries
        successfully. Public-pass cases test the limits of available verification.

        This is synthetic development data, not a representative hidden-final set.
        All variants of a family must stay together in CV. Do not inspect test gold
        before freezing the predictor, policy, caps and evaluation protocol.
        Preserve the input and evaluator hashes. Never overwrite measured suites.
    '''),encoding="utf-8")


def pytest_check(work,paths):
    env = {k:os.environ[k] for k in ("PATH","SYSTEMROOT","WINDIR","TEMP","TMP") if k in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",PYTHONPATH=str(work))
    cmd = [sys.executable,"-B","-m","pytest","-q","-p","no:cacheprovider",*paths]
    result = subprocess.run(cmd,cwd=work,env=env,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=20)
    return result.returncode == 0,result.stdout+result.stderr


def validate(output,report_path):
    manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8-sig"))
    oracle = json.loads((output/"oracle.json").read_text(encoding="utf-8-sig"))
    metadata = json.loads((output/"research_metadata.json").read_text(encoding="utf-8-sig"))
    assert len(manifest["tasks"]) == len(oracle) == len(metadata) == 144
    assert digest(dumps(oracle)) == manifest["oracle_sha256"]
    assert digest(dumps(metadata)) == manifest["research_metadata_sha256"]
    splits = collections.Counter(); domains = collections.Counter(); negative = collections.Counter(); groups = {}
    result = []; instructions = []
    for rec in manifest["tasks"]:
        item = oracle[rec["id"]]; meta = metadata[rec["id"]]; work = output/rec["task_path"]/"workspace"
        task = json.loads((work.parent/"task.json").read_text(encoding="utf-8-sig"))
        instructions.append(task["instruction"])
        assert set(task) == {"task_id","task_type","reward","instruction"}
        assert not ({"split","group","family","answerable","expected"} & set(rec))
        assert digest(dumps(task)) == rec["task_sha256"]
        actual = {p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file()}
        assert actual == set(rec["files_sha256"])
        files = {name:(work/name).read_text(encoding="utf-8") for name in actual}
        assert all(digest(body)==rec["files_sha256"][name] for name,body in files.items())
        split,domain = meta["split"],rec["domain"]
        splits[split] += 1; domains[split,domain] += 1; negative[split,domain] += not item["answerable"]
        groups.setdefault(meta["group"],set()).add(split)
        kind = item["validation_kind"]
        if kind == "math":
            if item["answerable"]: assert scalar(math_eval(item["expression"],item["variables"])) == item["expected"]
            else:
                values = [math_eval(item["expression"],world) for world in item["worlds"]]
                assert values == item["witness_answers"] and len(set(values)) == 2
        elif kind == "search":
            answers = search_answer(item["search_kind"],json.loads(files[item["data_file"]]))
            assert answers == item["candidates"]
            assert (len(answers)==1) == item["answerable"]
            if item["answerable"]: assert answers[0] == item["expected"]
            if item.get("witness_worlds"):
                witness_answers = [search_answer(item["search_kind"],world) for world in item["witness_worlds"]]
                assert all(len(values)==1 for values in witness_answers)
                assert witness_answers[0] != witness_answers[1]
        elif kind == "sql":
            if item["answerable"]: assert sql_answer(files,item["tables"],item["query"]) == item["expected"]
            else:
                answers = [sql_answer(world,item["tables"],item["query"]) for world in item["world_files"]]
                assert answers == item["witness_answers"] and len(set(answers)) == 2
                # Every observed cell agrees in both completions; only blank cells may differ.
                for name,body in files.items():
                    observed = list(csv.reader(io.StringIO(body)))
                    for world in item["world_files"]:
                        completed = list(csv.reader(io.StringIO(world[name])))
                        assert len(observed) == len(completed)
                        assert all(len(a)==len(b) and all(x==y or x=="" for x,y in zip(a,b)) for a,b in zip(observed,completed))
        else:
            assert not set(item["hidden_files"]) & set(files)
            public = sorted(name for name in files if Path(name).name.startswith("test_"))
            with tempfile.TemporaryDirectory(prefix="policy-check-") as tmp:
                sandbox = Path(tmp)/"work"; shutil.copytree(work,sandbox)
                initial,_ = pytest_check(sandbox,public)
                assert initial == (item["public_profile"] == "public_pass_hidden_fail"),rec["id"]
                for name,body in item["hidden_files"].items(): (sandbox/safe(name)).write_text(body,encoding="utf-8")
                hidden = sorted(item["hidden_files"])
                initial_hidden,_ = pytest_check(sandbox,hidden)
                assert not initial_hidden,rec["id"]
                if item["answerable"]:
                    for name,body in item["reference_files"].items(): (sandbox/safe(name)).write_text(body,encoding="utf-8")
                    passed,detail = pytest_check(sandbox,public+hidden)
                    assert passed,(rec["id"],detail)
                else:
                    assert item["proof_type"] and item["reason"]
        result.append({"id":rec["id"],"passed":True})
    assert splits == {"train":64,"calibration":32,"test":48}
    assert len(set(instructions)) == 144
    assert all(i==j or a not in b for i,a in enumerate(instructions) for j,b in enumerate(instructions))
    for split,total in (("train",16),("calibration",8),("test",12)):
        for domain in ("math","coding","search","data_analysis"):
            assert domains[split,domain] == total and negative[split,domain] == total//4
    assert len(groups) == 36 and all(len(s)==1 for s in groups.values())
    report = dict(passed=144,total=144,api_calls=0,network="none",split_counts=dict(splits),
        negative_count=sum(negative.values()),family_count=len(groups),cases=result,
        manifest_sha256=hashlib.sha256((output/"manifest.json").read_bytes()).hexdigest())
    report_path.write_text(dumps(report),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="cases"}))


def docker_validate(output,image):
    output = output.resolve()
    command = ["docker","run","--rm","--network","none","--read-only","--tmpfs","/tmp:rw,size=512m",
        "--cpus","2","--memory","2g","--pids-limit","128","--cap-drop","ALL","--security-opt","no-new-privileges",
        "--mount",f"type=bind,source={output},target=/suite,readonly",
        "--mount",f"type=bind,source={output},target=/out",
        "--mount",f"type=bind,source={Path(__file__).resolve()},target=/validator.py,readonly",
        "--entrypoint","python",image,"-B","/validator.py","--validate-only","--inside-docker","--output","/suite","--report","/out/validation.json"]
    subprocess.run(command,check=True,timeout=600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--inside-docker",action="store_true")
    parser.add_argument("--report",type=Path)
    parser.add_argument("--image",default="nsrc-cpu-candidate:20261112")
    args = parser.parse_args()
    if not args.validate_only: write_suite(args.output)
    if args.inside_docker:
        validate(args.output,args.report or args.output/"validation.json")
    else:
        docker_validate(args.output,args.image)


if __name__ == "__main__": main()
