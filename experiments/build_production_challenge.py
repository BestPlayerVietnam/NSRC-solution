"""Independent, deterministic 24-case production structural-stress benchmark.

No model, external dataset, network or production solver is used. This small
purpose-built set measures structural coverage, not hidden-final accuracy.
Run with Python 3.11; a successful build validates every oracle locally.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "suite_production"
SEED = 20261112073


def text(source):
    return textwrap.dedent(source).strip() + "\n"


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def csv_text(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue()


def scalar(value):
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def cases():
    rng = random.Random(SEED)
    result = []

    def add(domain, name, instruction, files, expected, **oracle):
        result.append(dict(domain=domain, name=name, instruction=instruction,
                           files=files, expected=expected, oracle=oracle))

    numeric = " Return only the final integer. Use only the stated facts; if they cannot determine a unique answer, return UNSOLVABLE."
    stock, daily, days, packs, each = rng.randint(180, 240), rng.randint(3, 8), 4, 3, rng.randint(12, 19)
    expr = f"({stock}-{daily}*{days}+{packs}*{each})*2"
    add("math", "warehouse_compatible", f"A warehouse starts with {stock} boxes. It ships {daily} boxes per day for {days} days. It receives {packs} deliveries of {each} boxes each. The stock is doubled by merging with an identical warehouse. How many boxes are there now?" + numeric,
        {}, str((stock-daily*days+packs*each)*2), validation_expr=expr)
    stock, daily, each = rng.randint(400, 520), rng.randint(5, 11), rng.randint(15, 23)
    add("math", "warehouse_split_compatible", f"A warehouse starts with {stock} boxes. It receives 2 deliveries of {each} boxes each. It ships {daily} boxes per day for 3 days. The stock is split equally into 4 warehouses (rounded down). How many boxes does each warehouse receive?" + numeric,
        {}, str((stock+2*each-3*daily)//4), validation_expr=f"({stock}+2*{each}-3*{daily})//4")
    crates, per, damaged, reserve, donation = rng.randint(15, 22), rng.randint(11, 18), rng.randint(8, 15), 17, rng.randint(9, 14)
    add("math", "school_packs", f"A school opens {crates} cartons, each containing {per} pencils. It discards {damaged} damaged pencils, keeps {reserve} for demonstrations, and receives two donated bags of {donation} pencils each. It then makes complete student packs of 6 pencils, keeping any remainder. How many complete packs can it make?" + numeric,
        {}, str((crates*per-damaged-reserve+2*donation)//6), validation_expr=f"({crates}*{per}-{damaged}-{reserve}+2*{donation})//6")
    count_a, count_b, price_a, price_b = rng.randint(7, 12), rng.randint(5, 9), 175, 260
    refund, shipping = rng.randint(120, 180), 145
    revenue = count_a*price_a+count_b*price_b-refund
    add("math", "cents_fee_share", f"A workshop sells {count_a} kits at {price_a} cents each and {count_b} posters at {price_b} cents each. After refunding {refund} cents, it pays a service fee equal to 5 percent of the remaining revenue, rounded down to whole cents. It then pays {shipping} cents for shipping and divides the balance into 3 equal whole-cent shares, retaining any remainder. How many cents are in one share?" + numeric,
        {}, str((revenue-revenue*5//100-shipping)//3), validation_expr=f"({revenue}-({revenue}*5//100)-{shipping})//3")
    water, shift, rain, transfer = rng.randint(700, 900), rng.randint(25, 40), rng.randint(60, 90), 43
    wet = water-3*shift+rain
    add("math", "water_after_leak", f"A tank initially contains {water} liters. Three cleaning shifts each remove {shift} liters. Rain adds {rain} liters. A leak then loses one quarter of the water present at that moment, rounded down to whole liters. Another {transfer} liters are transferred out. The remaining water fills as many complete 8-liter cans as possible. How many complete cans are filled?" + numeric,
        {}, str((wet-wet//4-transfer)//8), validation_expr=f"({wet}-{wet}//4-{transfer})//8")
    removed, added, families = 7, rng.randint(10, 15), 4
    add("math", "unknown_initial_inventory", f"The initial number of cans in a pantry was not recorded. For 3 days it gives away {removed} cans a day, then receives 2 cartons of {added} cans and distributes 5 cans to each of {families} families. How many cans remain?" + numeric,
        {}, None, reason="The initial inventory is unspecified; two nonnegative inventories satisfy every fact and give different answers.",
        witness={"worlds": [{"initial": 100}, {"initial": 200}], "answers": [100-3*removed+2*added-5*families, 200-3*removed+2*added-5*families]},
        validation_exprs=[f"100-3*{removed}+2*{added}-5*{families}", f"200-3*{removed}+2*{added}-5*{families}"])

    code_rule = " Fix implementation files only; do not edit, replace or disable tests. Additional checks are withheld. Preserve the documented API. After a successful repair the answer is DONE. If the specification is contradictory, return UNSOLVABLE."
    public = text('''
        import unittest
        import mylib as m
        class TestPositive(unittest.TestCase):
            def test_mixed(self):
                self.assertEqual(m.sum_positive([-8, 3, 7]), 10)
        if __name__ == '__main__': unittest.main()
    ''')
    add("coding", "single_module_compatible", "Repair mylib.sum_positive(values): return the sum of strictly positive integers, ignoring zero and negative values. For an empty list return 0." + code_rule,
        {"mylib.py": "def sum_positive(values):\n    return sum(values)\n", "test_mylib.py": public}, "PASS_HIDDEN",
        reference_files={"mylib.py": "def sum_positive(values):\n    return sum(x for x in values if x > 0)\n"},
        hidden_files={"test_hidden_positive.py": "from mylib import sum_positive\ndef test_hidden():\n    assert sum_positive([]) == 0\n    assert sum_positive([-4, -2, 0]) == 0\n    assert sum_positive([9, -1, 2, 0, 9]) == 20\n"})
    files = {"billing/__init__.py": "from .core import invoice_total\n",
             "billing/core.py": "from .rules import apply_discount\ndef invoice_total(lines, discount_percent):\n    subtotal = sum(price * quantity for price, quantity in lines)\n    return apply_discount(subtotal, discount_percent)\n",
             "billing/rules.py": "def apply_discount(cents, percent):\n    return cents + cents * percent // 100\n",
             "tests/test_invoice.py": "from billing import invoice_total\ndef test_invoice():\n    assert invoice_total([(125, 4), (90, 2)], 10) == 612\n"}
    add("coding", "package_discount_pytest", "Repair billing.invoice_total(lines, discount_percent). Each line is (integer price in cents, integer quantity). Sum price times quantity, then subtract floor(subtotal * discount_percent / 100). Inputs are nonnegative; percent is between 0 and 100. Return integer cents. Public tests use pytest in tests/." + code_rule,
        files, "PASS_HIDDEN", reference_files={"billing/rules.py": "def apply_discount(cents, percent):\n    return cents - cents * percent // 100\n"},
        hidden_files={"test_hidden_invoice.py": "from billing import invoice_total\ndef test_hidden():\n    assert invoice_total([], 25) == 0\n    assert invoice_total([(101, 1)], 7) == 94\n    assert invoice_total([(33, 3)], 100) == 0\n    assert invoice_total([(19, 2)], 0) == 38\n"})
    files = {"textkit/__init__.py": "from .reports import word_counts\n",
             "textkit/normalize.py": "def words(text):\n    return text.lower().split()\n",
             "textkit/reports.py": "from collections import Counter\nfrom .normalize import words\ndef word_counts(text):\n    return dict(Counter(words(text)))\n",
             "tests/test_words.py": "from textkit import word_counts\ndef test_punctuation():\n    assert word_counts('Fox, fox dog.') == {'fox': 2, 'dog': 1}\n"}
    add("coding", "unicode_multifile_tokens", "Repair textkit.word_counts(text). A word is a maximal consecutive run of Unicode alphabetic characters (str.isalpha); all other characters, including digits, underscores and punctuation, separate words. Lowercase each complete word using str.lower and count occurrences. Empty input returns {}. Tests use pytest." + code_rule,
        files, "PASS_HIDDEN", reference_files={"textkit/normalize.py": text('''
            def words(text):
                parts = []
                current = []
                for char in text:
                    if char.isalpha():
                        current.append(char)
                    elif current:
                        parts.append(''.join(current).lower())
                        current = []
                if current:
                    parts.append(''.join(current).lower())
                return parts
        ''')}, hidden_files={"test_hidden_words.py": "from textkit import word_counts\ndef test_hidden():\n    assert word_counts('') == {}\n    assert word_counts('Кот, КОТ; дом2дом') == {'кот': 2, 'дом': 2}\n    assert word_counts('red_blue---RED') == {'red': 2, 'blue': 1}\n    assert word_counts('123_!?') == {}\n"})
    files = {"src/__init__.py": "", "src/timeblocks/__init__.py": "from .spans import merge_blocks\n",
             "src/timeblocks/spans.py": "def merge_blocks(blocks):\n    return sorted(blocks)\n",
             "tests/unit/test_spans.py": "from src.timeblocks import merge_blocks\ndef test_overlap():\n    assert merge_blocks([(5, 9), (1, 6)]) == [(1, 9)]\n"}
    add("coding", "nested_src_interval_pytest", "Repair src.timeblocks.merge_blocks(blocks), where every element is an integer pair (start, end) with start < end. Return sorted tuples representing their union: merge both overlaps and touching endpoints. Do not mutate the input. Empty input returns []. Public pytest tests are nested in tests/unit/." + code_rule,
        files, "PASS_HIDDEN", reference_files={"src/timeblocks/spans.py": text('''
            def merge_blocks(blocks):
                merged = []
                for start, end in sorted(blocks):
                    if merged and start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                    else:
                        merged.append((start, end))
                return merged
        ''')}, hidden_files={"test_hidden_spans.py": text('''
            from src.timeblocks import merge_blocks
            def test_hidden():
                source = [(4, 8), (-2, 1), (1, 4), (15, 19), (16, 18)]
                saved = list(source)
                assert merge_blocks(source) == [(-2, 8), (15, 19)]
                assert source == saved
                assert merge_blocks([]) == []
                assert merge_blocks([(1, 3), (1, 3)]) == [(1, 3)]
        ''')})
    files = {"mylib.py": "def moving_sums(values, width):\n    return [sum(values[:width])]\n",
             "test_mylib.py": text('''
                import unittest
                import mylib as m
                class TestWindows(unittest.TestCase):
                    def test_windows(self):
                        self.assertEqual(m.moving_sums([2, 5, 1, 4], 2), [7, 6, 5])
                if __name__ == '__main__': unittest.main()
             ''')}
    add("coding", "single_module_window_compatible", "Repair mylib.moving_sums(values, width): for positive integer width, return sums of all complete consecutive windows in their original order. If width exceeds list length, return []. Do not mutate values." + code_rule,
        files, "PASS_HIDDEN", reference_files={"mylib.py": "def moving_sums(values, width):\n    return [sum(values[i:i + width]) for i in range(len(values) - width + 1)]\n"},
        hidden_files={"test_hidden_windows.py": "from mylib import moving_sums\ndef test_hidden():\n    assert moving_sums([], 1) == []\n    assert moving_sums([3, -2, 8], 1) == [3, -2, 8]\n    assert moving_sums([1, 2], 3) == []\n    assert moving_sums([1, 2, 3], 3) == [6]\n"})
    k = rng.randint(20, 40)
    contradictory = text(f'''
        import unittest
        import mylib as m
        class TestContract(unittest.TestCase):
            def test_same_input(self):
                self.assertEqual(m.f({k}), {k + 1})
                self.assertEqual(m.f({k}), {k + 2})
                self.assertIs(type(m.f({k})), int)
        if __name__ == '__main__': unittest.main()
    ''')
    add("coding", "contradictory_builtin_int", f"All functions must be stateless and deterministic. Implement mylib.f(x), returning a built-in int. For x={k}, the result must equal both {k + 1} and {k + 2}. Neither tests nor Python's standard library may be changed." + code_rule,
        {"mylib.py": "def f(x):\n    return x\n", "test_mylib.py": contradictory}, None,
        reason="A stateless deterministic built-in int for one input cannot equal two distinct integers.",
        witness={"input": k, "required_equal_values": [k+1, k+2], "required_type": "builtins.int"},
        hidden_files={"test_hidden_contract.py": f"from mylib import f\ndef test_hidden():\n    value = f({k})\n    assert type(value) is int\n    assert value == {k+1}\n    assert value == {k+2}\n"})

    closed = " Use only the complete local knowledge base in workspace/kb; no external lookup or guessing. Return only the requested value, or UNSOLVABLE if it is not uniquely determined."
    born = rng.randint(1965, 1994)
    add("search", "born_compatible", "In what year was Nora Venn born?" + closed,
        {"kb/person_nora_venn.md": f"Name: Nora Venn\nBorn: {born}\nOccupation: cartographer\n"}, str(born), evidence=[{"path": "kb/person_nora_venn.md", "contains": f"Born: {born}"}])
    born = rng.randint(1959, 1989)
    files = {f"kb/{i:03}.txt": f"Archive record {i}: storage bay {i + 80} was inspected. This record concerns neither vessels nor people.\n" for i in range(31)}
    files["kb/031.txt"] = "The research vessel Cobalt Finch is owned by Mara Vale.\n"
    files["kb/032.txt"] = f"Mara Vale was born in {born}. Her residence is not recorded.\n"
    add("search", "neutral_files_two_hop", "What is the birth year of the owner of the research vessel Cobalt Finch?" + closed, files, str(born),
        evidence=[{"path": "kb/031.txt", "contains": "owned by Mara Vale"}, {"path": "kb/032.txt", "contains": f"born in {born}"}])
    registry = f"RV-{rng.randint(300, 899)}"
    archive = "".join(f"Record {i:03}: coastal sensor {i+1000} passed inspection; archive shelf B was cleaned and relabeled.\n" for i in range(110))
    archive += f"Final registry correction: the official registry code of station Sable Reach is {registry}.\n"
    add("search", "late_evidence_long_file", "What is the official registry code of station Sable Reach?" + closed,
        {"kb/record_17.txt": archive, "kb/index.txt": "Registry corrections are recorded after the inspection entries.\n"}, registry,
        evidence=[{"path": "kb/record_17.txt", "contains": f"Sable Reach is {registry}", "minimum_offset": 8500}])
    files = {"kb/record_a.txt": "Leah Dorne coordinates the project Blue Lantern.\n",
             "kb/record_b.txt": "The project Blue Lantern is operated by Meridian Lab.\n",
             "kb/record_c.txt": "Meridian Lab is located in the city of Orlovsk.\n",
             "kb/record_d.txt": "Leah Dorne previously visited Belorechye; that city is not the location of her project's laboratory.\n"}
    add("search", "neutral_three_hop", "In which city is the laboratory operating the project coordinated by Leah Dorne?" + closed, files, "Orlovsk",
        evidence=[{"path": "kb/record_a.txt", "contains": "Blue Lantern"}, {"path": "kb/record_b.txt", "contains": "Meridian Lab"}, {"path": "kb/record_c.txt", "contains": "Orlovsk"}])
    add("search", "headquarters_compatible", "Where is the headquarters of Evan Reed's employer?" + closed,
        {"kb/person_evan_reed.md": "Name: Evan Reed\nEmployer: Cedar Forge\n",
         "kb/company_cedar_forge.md": "Company: Cedar Forge\nHeadquarters: Velinsk\n"}, "Velinsk",
        evidence=[{"path": "kb/person_evan_reed.md", "contains": "Employer: Cedar Forge"}, {"path": "kb/company_cedar_forge.md", "contains": "Headquarters: Velinsk"}])
    add("search", "birth_not_recorded", "What is the birth year of Ilya Soren?" + closed,
        {"kb/record_08.txt": "Ilya Soren is an archivist at North Pier. The archive records no dates concerning his life.\n",
         "kb/record_12.txt": "North Pier maintains marine charts and equipment inventories.\n"}, None,
        reason="The complete allowed KB contains no birth date or facts implying one; both witness years are consistent.",
        witness={"worlds": [{"birth_year": 1974}, {"birth_year": 1983}], "answers": [1974, 1983]},
        forbidden_fact_pattern=r"\b(?:19|20)\d{2}\b")

    data_rule = " Interpret numeric columns as numbers, not lexicographic strings. Use only the provided CSV files. Return only the final integer, or UNSOLVABLE if the files do not determine a unique answer."
    rows = [(p, m, rng.randint(2, 14)) for m in range(1, 9) for p in ("bolt", "nut")]
    add("data_analysis", "monthly_compatible", "How many units of product 'bolt' were sold in months 2 through 6 inclusive?" + data_rule,
        {"sales.csv": csv_text(["product", "month", "units"], rows)}, str(sum(u for p,m,u in rows if p == "bolt" and 2 <= m <= 6)),
        tables={"sales": {"file": "sales.csv", "types": ["TEXT", "INTEGER", "INTEGER"]}},
        validation_sql="SELECT SUM(units) FROM sales WHERE product='bolt' AND month BETWEEN 2 AND 6")
    rows = [("Harbor torch" if i%2 else "North lamp", rng.randint(2, 13), rng.randint(70, 140)) for i in range(12)]
    add("data_analysis", "spaces_and_quoted_threshold", "In daily sales.csv, sum Units Sold times Unit Price for rows whose Product name is Harbor torch and whose Units Sold is at least '7' (a numeric threshold)." + data_rule,
        {"daily sales.csv": csv_text(["Product name", "Units Sold", "Unit Price"], rows)}, str(sum(q*p for name,q,p in rows if name == "Harbor torch" and q >= 7)),
        tables={"daily_sales": {"file": "daily sales.csv", "types": ["TEXT", "INTEGER", "INTEGER"]}},
        validation_sql='SELECT COALESCE(SUM("Units Sold" * "Unit Price"),0) FROM daily_sales WHERE "Product name"=\'Harbor torch\' AND "Units Sold">=\'7\'')
    rows = [("Мёд" if i%3 else "Чай", "Восток" if i%2 else "Запад", rng.randint(2, 9), rng.randint(100, 230)) for i in range(14)]
    add("data_analysis", "cyrillic_columns", "В файле продажи.csv найдите общую выручку: сумму Количество * Цена для строк, где Товар равен Мёд, Регион равен Восток и Количество не меньше 4." + data_rule,
        {"продажи.csv": csv_text(["Товар", "Регион", "Количество", "Цена"], rows)}, str(sum(q*p for product,region,q,p in rows if product == "Мёд" and region == "Восток" and q >= 4)),
        tables={"sales": {"file": "продажи.csv", "types": ["TEXT", "TEXT", "INTEGER", "INTEGER"]}},
        validation_sql='SELECT COALESCE(SUM("Количество"*"Цена"),0) FROM sales WHERE "Товар"=\'Мёд\' AND "Регион"=\'Восток\' AND "Количество">=4')
    customers = [(1, "Perm"), (2, "Omsk"), (3, "Perm"), (4, "Tomsk")]
    orders = [(i+1, i%4+1, rng.randint(15, 80), "paid" if i%3 else "pending") for i in range(16)]
    add("data_analysis", "two_csv_join", "Join orders.csv.customer_id to customers.csv.id. What is the sum of amount for paid orders belonging to customers whose city is Perm? A customer can have multiple orders; count each qualifying order once." + data_rule,
        {"customers.csv": csv_text(["id", "city"], customers), "orders.csv": csv_text(["id", "customer_id", "amount", "status"], orders)},
        str(sum(amount for _,customer,amount,status in orders if customer in {1,3} and status == "paid")),
        tables={"customers": {"file": "customers.csv", "types": ["INTEGER", "TEXT"]}, "orders": {"file": "orders.csv", "types": ["INTEGER", "INTEGER", "INTEGER", "TEXT"]}},
        validation_sql="SELECT SUM(o.amount) FROM orders o JOIN customers c ON o.customer_id=c.id WHERE o.status='paid' AND c.city='Perm'")
    rows = [("bolts" if i%2 else "nuts", "East" if i%3 else "West", rng.randint(3, 17)) for i in range(15)]
    add("data_analysis", "region_compatible", "How many units of bolts were sold in the East region?" + data_rule,
        {"shipments.csv": csv_text(["product", "region", "units"], rows)}, str(sum(q for p,r,q in rows if p == "bolts" and r == "East")),
        tables={"shipments": {"file": "shipments.csv", "types": ["TEXT", "TEXT", "INTEGER"]}},
        validation_sql="SELECT SUM(units) FROM shipments WHERE product='bolts' AND region='East'")
    rows = [(1, 71, 35, "paid"), (2, 82, 49, "paid"), (3, 71, 17, "pending")]
    add("data_analysis", "missing_customer_mapping", "What is the total amount of paid orders for customers based in Perm? Customer identifiers are arbitrary identifiers and do not encode location. The only supplied dataset is orders.csv." + data_rule,
        {"orders.csv": csv_text(["id", "customer_id", "amount", "status"], rows)}, None,
        reason="No supplied column or file maps customer IDs to cities. Two assignments of city satisfy all observed records and give different totals.",
        witness={"worlds": [{"71": "Perm", "82": "Perm"}, {"71": "Omsk", "82": "Omsk"}], "answers": [84, 0]},
        validation_missing_mapping={"file": "orders.csv", "city": "Perm"})
    return result


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or ":" in name:
        raise ValueError("unsafe generated path: " + name)
    return path


def build(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen suite: {output}; use --validate-only")
    items = cases()
    rewards = random.Random(SEED ^ 0xBEEF)
    records, oracle = [], {}
    for item in sorted(items, key=lambda value: digest(value["name"])):
        ident = "prod_" + item["domain"] + "_" + digest(str(SEED)+item["name"])[:10]
        task = dict(task_id=ident, task_type=item["domain"], reward=rewards.randint(1,3), instruction=item["instruction"])
        folder = output / "tasks" / ident
        workspace = folder / "workspace"
        workspace.mkdir(parents=True)
        (folder / "task.json").write_text(dumps(task), encoding="utf-8")
        for filename, content in item["files"].items():
            path = workspace / safe_relative(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        records.append(dict(id=ident, domain=item["domain"], reward=task["reward"], task_path="tasks/"+ident,
                            task_sha256=digest(dumps(task)), files_sha256={name:digest(content) for name,content in item["files"].items()}))
        oracle[ident] = {"expected": item["expected"], "answerable": item["expected"] is not None,
                         "scenario": item["name"], "hidden_files": {}, "reason": None,
                         **item["oracle"]}
    manifest = dict(version=1, seed=SEED, label="structural-stress", task_count=24,
                    description="Paired production structural coverage check; not a representative estimate of hidden-final accuracy.",
                    reward_policy="Independent seeded integers 1..3, not tied to difficulty. This is not a 100-task routing evaluation.",
                    oracle_sha256=digest(dumps(oracle)), tasks=records)
    (output / "oracle.json").write_text(dumps(oracle), encoding="utf-8")
    (output / "manifest.json").write_text(dumps(manifest), encoding="utf-8")
    (output / "README.md").write_text(text('''
        # Production structural-stress suite
        24 authored synthetic tasks, six per domain; exactly one unanswerable case
        in every domain. Some cases retain the original simple layout; others
        exercise nested/multiple code files, pytest-style functions, CSV joins,
        Cyrillic/spaced columns, numeric thresholds, late evidence, neutral file
        names and multi-hop lookup. Fixed seed; independently assigned rewards 1..3.

        Only tasks/<id>/task.json and tasks/<id>/workspace belong to the solver.
        oracle.json, manifest metadata and validation.json are evaluator-only.
        Hidden tests and reference implementations are embedded in oracle.json,
        never placed in solver workspaces. Public tests contain normal examples.
        Run each policy on a fresh copy of tasks/. Do not tune on benchmark answers.
        Report paired correctness, refusals, cost and incomplete coverage. This
        small structural-stress suite is not a representative hidden-final sample
        and is not an unbiased estimate after inspecting its outcomes for tuning.
    '''), encoding="utf-8")
    return manifest, oracle


CHECK_CODE = text('''
    import inspect, pathlib, runpy, sys, unittest
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(pathlib.Path.cwd()))
    for filename in sys.argv[1:]:
        namespace = runpy.run_path(filename)
        suite = unittest.TestSuite()
        for name, obj in namespace.items():
            if inspect.isclass(obj) and issubclass(obj, unittest.TestCase):
                suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(obj))
            elif name.startswith('test_') and inspect.isfunction(obj):
                obj()
        if not unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful():
            raise SystemExit(1)
''')


def run_checks(workspace, filenames):
    completed = subprocess.run([sys.executable, "-B", "-c", CHECK_CODE, *filenames], cwd=workspace,
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
    return completed.returncode == 0, completed.stdout + completed.stderr


def validate(output, *, write_report=True):
    import re
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    oracle = json.loads((output / "oracle.json").read_text(encoding="utf-8"))
    assert digest(dumps(oracle)) == manifest["oracle_sha256"]
    assert len(manifest["tasks"]) == len(oracle) == 24
    summaries = []
    for rec in manifest["tasks"]:
        folder = output / rec["task_path"]
        work = folder / "workspace"
        task = json.loads((folder / "task.json").read_text(encoding="utf-8"))
        item = oracle[rec["id"]]
        assert "expected" not in task and "answerable" not in task and "oracle" not in task
        assert digest(dumps(task)) == rec["task_sha256"]
        assert set(str(p.relative_to(work)).replace("\\", "/") for p in work.rglob("*") if p.is_file()) == set(rec["files_sha256"])
        for name, expected_hash in rec["files_sha256"].items():
            assert digest((work/name).read_text(encoding="utf-8")) == expected_hash, name
        assert not (set(item.get("hidden_files", {})) & set(rec["files_sha256"]))
        checks = []
        if "validation_expr" in item:
            actual = eval(item["validation_expr"], {"__builtins__": {}}, {})
            assert scalar(actual) == item["expected"]
            checks.append("independent arithmetic expression")
        if "validation_exprs" in item:
            answers = [eval(expr, {"__builtins__": {}}, {}) for expr in item["validation_exprs"]]
            assert answers == item["witness"]["answers"] and len(set(answers)) == 2
            checks.append("two consistent numeric worlds")
        if "evidence" in item:
            for ev in item["evidence"]:
                body = (work/ev["path"]).read_text(encoding="utf-8")
                assert ev["contains"] in body
                assert body.index(ev["contains"]) >= ev.get("minimum_offset", 0)
            checks.append("complete evidence chain present")
        if "forbidden_fact_pattern" in item:
            assert not any(re.search(item["forbidden_fact_pattern"], p.read_text(encoding="utf-8")) for p in work.rglob("*") if p.is_file())
            assert len(set(item["witness"]["answers"])) == 2
            checks.append("two KB-compatible birth years")
        if "validation_sql" in item:
            with sqlite3.connect(":memory:") as con:
                for table, spec in item["tables"].items():
                    with (work/spec["file"]).open(encoding="utf-8", newline="") as stream:
                        rows = list(csv.reader(stream))
                    quote = lambda value: '"' + value.replace('"', '""') + '"'
                    fields = ",".join(quote(col)+" "+kind for col,kind in zip(rows[0], spec["types"]))
                    con.execute("CREATE TABLE "+quote(table)+" ("+fields+")")
                    con.executemany("INSERT INTO "+quote(table)+" VALUES ("+",".join("?" for _ in rows[0])+")", rows[1:])
                actual = con.execute(item["validation_sql"]).fetchone()[0]
                assert scalar(actual) == item["expected"], (rec["id"], actual, item["expected"])
            checks.append("independent typed SQLite query")
        if "validation_missing_mapping" in item:
            spec = item["validation_missing_mapping"]
            with (work/spec["file"]).open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                assert "city" not in reader.fieldnames
                rows = list(reader)
            answers = [sum(int(row["amount"]) for row in rows if row["status"] == "paid" and world[row["customer_id"]] == spec["city"]) for world in item["witness"]["worlds"]]
            assert answers == item["witness"]["answers"] and len(set(answers)) == 2
            checks.append("two compatible missing-join worlds")
        if rec["domain"] == "coding":
            public = sorted(str(p.relative_to(work)).replace("\\", "/") for p in work.rglob("test_*.py"))
            assert public
            with tempfile.TemporaryDirectory(prefix="nsrc_structural_") as temp:
                sandbox = Path(temp) / "workspace"
                shutil.copytree(work, sandbox)
                broken_passes, _ = run_checks(sandbox, public)
                assert not broken_passes, "broken implementation unexpectedly passes public examples"
                if item["answerable"]:
                    for name, source in item["reference_files"].items():
                        path = sandbox/safe_relative(name)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(source, encoding="utf-8")
                    hidden = []
                    for name, source in item["hidden_files"].items():
                        path = sandbox/"_evaluator_hidden"/safe_relative(name)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(source, encoding="utf-8")
                        hidden.append(str(path.relative_to(sandbox)))
                    passed, detail = run_checks(sandbox, public+hidden)
                    assert passed, (rec["id"], detail)
                    checks.append("reference passes public and separate hidden checks; original fails")
                else:
                    witness = item["witness"]
                    assert witness["required_type"] == "builtins.int"
                    assert len(set(witness["required_equal_values"])) == 2
                    checks.append("incompatible equalities on deterministic built-in integer")
        assert checks, rec["id"]
        summaries.append(dict(id=rec["id"], answerable=item["answerable"], passed=True, checks=checks))
    for domain in ("math", "coding", "search", "data_analysis"):
        ids = [rec["id"] for rec in manifest["tasks"] if rec["domain"] == domain]
        assert len(ids) == 6 and sum(not oracle[ident]["answerable"] for ident in ids) == 1
    report = dict(passed=24, total=24, seed=SEED, api_calls=0, label="structural-stress", cases=summaries)
    if write_report:
        (output/"validation.json").write_text(dumps(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        build(args.output)
    report = validate(args.output)
    print(json.dumps({"suite": str(args.output), "passed": report["passed"], "total": report["total"], "api_calls": 0}))


if __name__ == "__main__":
    main()
