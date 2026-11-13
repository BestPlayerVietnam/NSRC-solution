"""Offline satisfiable counterexamples. Run: python -B experiments/eblo_counterexamples.py.

Imports only the SHA-pinned EBLO AST scanner in a read-only child process.
Fixture Python files are parsed, never imported or executed. No model/API calls.
"""
from __future__ import annotations
import argparse
import ast
import csv
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCANNER = "tmp_repo-master/harness/task_scanner.py"
FILTER = "tmp_repo-master/harness/unfeasible_filter.py"
SCANNER_SHA256 = "bfcac0b4acce94dcb96752a1ebdc3344806d563f1068231b5f8c18b8107d7099"
CASES = [
    {
        "id": "compatible_bool_numeric_equality", "domain": "coding",
        "instruction": "Implement a deterministic, stateless function f(x). The return value must be a built-in int. Return 1 for every x.",
        "files": {
            "mylib.py": "def f(x):\n    return 1\n",
            "test_mylib.py": "from mylib import f\n\ndef test_equalities():\n    assert f(5) == 1\n    assert f(5) == True\n",
        },
        "trigger": "contradictory_asserts",
    },
    {
        "id": "mutable_stateful_pop", "domain": "coding",
        "instruction": "Implement f(items) as a stateful operation: remove and return the first item from the supplied mutable list, updating that list. The return value must be a built-in int.",
        "files": {
            "mylib.py": "def f(items):\n    return items.pop(0)\n",
            "test_mylib.py": "from mylib import f\n\ndef test_pop():\n    items = [10, 20]\n    assert f(items) == 10\n    assert f(items) == 20\n    assert items == []\n",
        },
        "trigger": "contradictory_asserts",
    },
    {
        "id": "empty_csv_count_zero", "domain": "data_analysis",
        "instruction": "Count the number of data rows in sales.csv. Return 0 when the table has no data rows.",
        "files": {"sales.csv": "product,units\n"},
        "trigger": "empty_csv",
    },
    {
        "id": "quoted_value_is_not_column", "domain": "data_analysis",
        "instruction": "Sum units in sales.csv where product equals 'sku_42'.",
        "files": {"sales.csv": "product,units\nsku_42,3\nsku_43,99\nsku_42,4\n"},
        "trigger": "missing_requested_columns",
    },
]
SOURCE_FILES = (
    SCANNER, FILTER, "feasibility.py", "TASK/Регламент соревнования.md",
    "tmp_repo-master/harness/jiuwen_agent.py",
    "tmp_repo-master/harness/budget_ledger.py",
    "tmp_repo-master/harness/estimator.py",
    "tmp_repo-master/eblo/competition_case/fitness.py",
    "tmp_repo-master/eblo/competition_case/tournament_sim.py",
    "tmp_repo-master/tests/test_full_100_sim.py",
)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def protected_hashes():
    paths = sorted(ROOT.glob("*.py"))
    if (ROOT / "Dockerfile").is_file():
        paths.append(ROOT / "Dockerfile")
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}

def reference_witness(case):
    """Independent authored stdlib logic; never eval/import fixture source."""
    ident = case["id"]
    if ident == "compatible_bool_numeric_equality":
        def constant_one(_):
            return 1
        result = [constant_one(5), constant_one(5)]
        checks = [all(type(v) is int for v in result), result[0] == 1, result[1] == True]
        reason = "Built-in int 1 compares equal to both 1 and True."
    elif ident == "mutable_stateful_pop":
        items = [10, 20]
        def pop_first(values):
            return values.pop(0)
        values = [pop_first(items), pop_first(items)]
        checks = [values == [10, 20], items == [], all(type(v) is int for v in values)]
        result = {"returns": values, "remaining": items}
        reason = "The same list changes between calls; pop returns 10, then 20."
    elif ident == "empty_csv_count_zero":
        result = len(list(csv.DictReader(io.StringIO(case["files"]["sales.csv"]))))
        checks = [result == 0]
        reason = "A header-only CSV has a well-defined row count of zero."
    else:
        rows = list(csv.DictReader(io.StringIO(case["files"]["sales.csv"])))
        result = sum(int(r["units"]) for r in rows if r["product"] == "sku_42")
        checks = [result == 7, set(rows[0]) == {"product", "units"}]
        reason = "'sku_42' is a cell value, not a requested column."
    if not all(checks):
        raise AssertionError("Reference witness failed: " + ident)
    return {"satisfiable": True, "expected_answer": result,
            "checks_passed": len(checks), "explanation": reason,
            "fixture_code_executed": False}

def verified_hard_rules():
    """Check actual hard-return AST branches without importing the filter."""
    tree = ast.parse((ROOT / FILTER).read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "evaluate_tier0_static")
    wanted = {
        "contradictory_asserts": "tf.contradictory_asserts",
        "missing_requested_columns": "tf.missing_requested_columns",
        "empty_csv": "tf.domain == 'data_analysis' and tf.csv_row_count == 0",
    }
    found = {}
    for label, expression in wanted.items():
        expected = ast.dump(ast.parse(expression, mode="eval").body)
        node = next((n for n in method.body if isinstance(n, ast.If)
                     and ast.dump(n.test) == expected), None)
        if node is None:
            raise AssertionError("EBLO hard rule changed: " + label)
        returns = [n for n in node.body if isinstance(n, ast.Return)]
        if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
            raise AssertionError("Unexpected filter body")
        call = returns[0].value
        kw = {k.arg: k.value for k in call.keywords}
        flag, tier = kw.get("is_feasible"), kw.get("tier_triggered")
        if not (isinstance(call.func, ast.Name) and call.func.id == "FilterDecision"
                and isinstance(flag, ast.Constant) and flag.value is False
                and isinstance(tier, ast.Constant) and tier.value == 0):
            raise AssertionError("Rule no longer hard-rejects")
        found[label] = {"path": FILTER, "line": node.lineno,
                        "condition": expression, "tier": 0, "is_feasible": False,
                        "validation": "Actual filter AST; filter not imported",
                        "qualifying_mode": False}
    return found

def readonly_audit(event, args):
    """Extra guard for trusted pinned code, not an OS security sandbox."""
    if (event.startswith("socket.") or event.startswith("subprocess.")
            or event in {"os.system", "os.posix_spawn", "os.exec", "os.spawn",
                         "os.remove", "os.rmdir", "os.mkdir", "os.rename",
                         "os.link", "os.symlink", "os.truncate"}):
        raise RuntimeError("Diagnostic worker forbids " + event)
    if event == "open":
        _, mode, flags = args
        writing = isinstance(mode, str) and any(c in mode for c in "wax+")
        mask = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if writing or (isinstance(flags, int) and flags & mask):
            raise RuntimeError("Diagnostic worker forbids file writes")

def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def worker(fixture_root):
    if sha256(ROOT / SCANNER) != SCANNER_SHA256:
        raise RuntimeError("Scanner changed: inspect before repinning SHA")
    tree = ast.parse((ROOT / SCANNER).read_text(encoding="utf-8"))
    allowed = {"__future__", "ast", "csv", "json", "os", "re",
               "dataclasses", "pathlib", "typing"}
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
    if not imports <= allowed:
        raise RuntimeError("Unexpected scanner dependency")
    sys.addaudithook(readonly_audit)
    scanner_module = import_file(ROOT / SCANNER, "_eblo_ast_scanner_diagnostic")
    own_module = import_file(ROOT / "feasibility.py", "_our_feasibility_diagnostic")
    scanner = scanner_module.ZeroTokenTaskScanner()
    results = []
    fields = ("domain", "contradictory_asserts", "has_syntax_error_in_tests",
              "csv_row_count", "csv_columns", "missing_requested_columns")
    for case in CASES:
        task_dir = fixture_root / case["id"]
        features = scanner.scan_single_task(task_dir, task_dir / "task.json")
        if features is None:
            raise AssertionError("Scanner failed to read fixture")
        own = own_module.analyze_task(
            {"id": case["id"], "task_type": case["domain"],
             "instruction": case["instruction"]}, task_dir / "workspace")
        results.append({"id": case["id"],
                        "eblo_features": {n: getattr(features, n) for n in fields},
                        "our_feasibility": dataclasses.asdict(own)})
    forbidden = sorted(n for n in sys.modules if n == "harness"
                       or n.startswith("harness.")
                       or n.split(".")[0] in {"openjiuwen", "httpx", "openai"})
    if forbidden:
        raise AssertionError("Unexpected runtime imports: " + repr(forbidden))
    print(json.dumps({"results": results, "unexpected_runtime_imports": forbidden,
                      "scanner_imports": sorted(imports)},
                     ensure_ascii=False, default=str))

def link(relative, line):
    return f"[{relative}:{line}](<{(ROOT / relative).as_posix()}:{line}>)"

def review_text(report):
    rows = "\n".join(
        f"| {r['id']} | {json.dumps(r['witness']['expected_answer'], ensure_ascii=False)} "
        f"| {r['trigger']} | Нет |" for r in report["cases"])
    return f"""# EBLO: воспроизводимый офлайн review

Воспроизведение: python -B experiments/eblo_counterexamples.py.
Отчёт: {link("experiments/eblo_counterexamples.json", 1)}.

Подтверждены **4/4 ложных hard-flags на решаемых fixtures**.
Это целевые контрпримеры, не оценка частоты ошибок на конкурсе.
Решения проверила независимая stdlib-логика; fixture Python-файлы не исполнялись.
Импортирован только SHA-pinned AST-сканер EBLO и наш feasibility.py.
Child: -I -B, временный cwd, env без ключей, audit-запрет сети, подпроцессов
и записей. Это guard прочитанного кода, не OS sandbox. Вызовов LLM: **0**.

| Fixture | Проверенный результат | Hard-trigger EBLO | Наш hard skip |
|---|---|---|---|
{rows}

**Причины.** Текст RHS различает 1 и True, хотя Python equality совместима.
Сканер не учитывает изменение list между двумя f(items):
{link(SCANNER, 246)}, hard-ветка {link(FILTER, 54)}.
Пустая таблица имеет COUNT=0, но отклоняется: {link(FILTER, 68)}.
Quoted token с underscore принимается за колонку, включая значение sku_42:
{link(SCANNER, 315)}, {link(FILTER, 61)}.
Hard-ветки подтверждены разбором AST, filter/runtime не импортировался.
Сравнение с нашим кодом касается только analyze_task, не всего planner.

**Цель — лексикографическая.** Регламент требует максимум баллов,
при равенстве — минимум токенов: {link("TASK/Регламент соревнования.md", 123)}.
EBLO использует reward - 0.01 * tokens, называя это official:
{link("tmp_repo-master/eblo/competition_case/fitness.py", 22)}.
100 баллов за 10000 токенов лучше 99 баллов бесплатно по регламенту
(оба в бюджете 100000), но scalar fitness EBLO предпочитает второй.
В наших экспериментах сравнивать (reward, -weighted_tokens) внутри одного
fixed-model pipeline с лимитом 100000. Число независимых экспериментов
не ограничивается этим лимитом. Переключение моделей разрешено исходным
регламентом, однако исключено в наших контролируемых сравнениях.

**Mock cap не моделирует truncation.**
{link("tmp_repo-master/eblo/competition_case/tournament_sim.py", 217)}
уменьшает completion_tokens, но строка 226 возвращает полный oracle-code,
независимо от cap; reasoning добавляется поверх cap (строка 223).
Поиск политики может награждать маленький cap без реальной потери качества.
Тест использует accuracy_scale=1e9 и ground truth:
{link("tmp_repo-master/tests/test_full_100_sim.py", 54)}.
Это проверка инфраструктуры, не доказательство качества или стоимости модели.

**Учёт расхода.** Guard проверяет превышение после вызова, без резерва:
{link("tmp_repo-master/harness/budget_ledger.py", 86)}.
Если record_llm_call бросает исключение, следующий log не выполняется:
{link("tmp_repo-master/harness/jiuwen_agent.py", 298)}.
Сумма completion + reasoning требует подтверждения семантики usage —
reasoning может уже входить в completion. Название p90 не даёт калибровки:
{link("tmp_repo-master/harness/estimator.py", 175)} задаёт множитель.

Переносить стоит проверку первой дешёвой попытки, остановку повтора одинаковой
ошибки и отбор контекста. Hard-skip по эвристическим маркерам, count-only
stagnation и обрезку середины каждого сообщения переносить не следует.
Runtime-файлы не изменены; SHA до/после включены в JSON.
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker.resolve())
        return
    before = protected_hashes()
    sources = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    if sources[SCANNER] != SCANNER_SHA256:
        raise RuntimeError("Refusing to import unreviewed scanner")
    rules = verified_hard_rules()
    witnesses = {case["id"]: reference_witness(case) for case in CASES}
    with tempfile.TemporaryDirectory(prefix="eblo_counterexamples_") as temporary:
        fixture_root = Path(temporary)
        for case in CASES:
            task_dir = fixture_root / case["id"]
            ws = task_dir / "workspace"
            ws.mkdir(parents=True)
            task = {"id": case["id"], "domain": case["domain"],
                    "task_type": case["domain"], "reward": 10,
                    "question": case["instruction"], "instruction": case["instruction"]}
            (task_dir / "task.json").write_text(
                json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
            for name, contents in case["files"].items():
                if name.endswith(".py"):
                    ast.parse(contents, filename=name)
                (ws / name).write_text(contents, encoding="utf-8", newline="\n")
        keys = ("SystemRoot", "WINDIR", "COMSPEC", "TEMP", "TMP",
                "PATH", "PATHEXT", "LANG", "LC_ALL")
        env = {k: os.environ[k] for k in keys if k in os.environ}
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
             "--worker", str(fixture_root)],
            cwd=fixture_root, env=env, text=True, encoding="utf-8",
            capture_output=True, timeout=30, check=False)
        if completed.returncode:
            raise RuntimeError("Read-only scanner worker failed:\n" + completed.stderr[-5000:])
        observed = json.loads(completed.stdout)
    after = protected_hashes()
    if before != after:
        raise RuntimeError("Protected runtime changed during diagnostic")
    if sources != {name: sha256(ROOT / name) for name in SOURCE_FILES}:
        raise RuntimeError("Reviewed source changed during diagnostic")
    by_id = {r["id"]: r for r in observed["results"]}
    results = []
    for case in CASES:
        entry = by_id[case["id"]]
        features, trigger = entry["eblo_features"], case["trigger"]
        triggered = (
            features["domain"] == "data_analysis" and features["csv_row_count"] == 0
            if trigger == "empty_csv" else bool(features[trigger]))
        if not triggered:
            raise AssertionError("EBLO flag no longer reproduced: " + case["id"])
        if entry["our_feasibility"]["hard_impossible"]:
            raise AssertionError("Our analyzer falsely rejected: " + case["id"])
        results.append({
            **entry, "domain": case["domain"], "instruction": case["instruction"],
            "fixture_sources": case["files"],
            "fixture_sha256": {k: hashlib.sha256(v.encode()).hexdigest()
                               for k, v in sorted(case["files"].items())},
            "trigger": trigger, "eblo_hard_rule": rules[trigger],
            "witness": witnesses[case["id"]], "false_hard_flag": True})
    report = {
        "schema": "eblo-counterexamples-v1", "api_calls": 0,
        "fixtures_executed": False, "filter_imported": False,
        "scope": "Targeted counterexamples, not a contest false-positive-rate estimate",
        "summary": {"cases": len(results), "reference_witnesses_passed": len(witnesses),
                    "false_hard_flags": sum(r["false_hard_flag"] for r in results),
                    "our_hard_impossible": sum(r["our_feasibility"]["hard_impossible"]
                                              for r in results)},
        "isolation": {"python_flags": ["-I", "-B"], "cwd": "temporary fixture directory",
                      "env": "OS allowlist; no API credentials",
                      "audit_blocks": ["network", "subprocess", "file writes"],
                      "os_sandbox_claimed": False},
        "source_sha256": sources, "runtime_before_sha256": before,
        "runtime_after_sha256": after, "runtime_unchanged": True,
        "scanner_imports": observed["scanner_imports"],
        "unexpected_runtime_imports": observed["unexpected_runtime_imports"],
        "cases": results}
    output = ROOT / "experiments/eblo_counterexamples.json"
    review = ROOT / "experiments/EBLO_REVIEW.md"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    review.write_text(review_text(report), encoding="utf-8", newline="\n")
    print(json.dumps({**report["summary"], "runtime_unchanged": True, "api_calls": 0,
                      "report": str(output), "review": str(review)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
