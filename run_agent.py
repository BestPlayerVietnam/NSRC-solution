"""Budget-aware openJiuwen contest harness.

The evaluator mounts writable tasks at --tasks-dir.  We deliberately ignore the
sample-only `expected` field and never use it to derive an answer.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def atomic_answer(task_dir: Path, workspace: Path, answer: str) -> None:
    """Persist immediately; graders have described both possible locations."""
    answer = str(answer).strip()
    if not answer or "\n" in answer:
        raise ValueError("answer must contain exactly one nonempty line")
    for folder in dict.fromkeys((workspace, task_dir)):
        folder.mkdir(parents=True, exist_ok=True)
        temp = folder / "answer.txt.tmp"
        temp.write_text(answer + "\n", encoding="utf-8")
        temp.replace(folder / "answer.txt")


def math_answer(instruction: str) -> str | None:
    """Exact integer interpreter for warehouse arithmetic in the sample family."""
    m = re.search(r"starts? with (\d+) boxes", instruction, re.I)
    if not m:
        return None
    stock = int(m.group(1))
    tail = instruction[m.end():]
    events: list[tuple[int, str, tuple[int, ...]]] = []
    matched_spans: list[tuple[int, int]] = []
    patterns = (
        ("split", r"split equally into (\d+) warehouses?[^.]*?\(rounded down\)"),
        ("double", r"(?:stock is )?doubled by merging"),
        ("ship", r"ships? (\d+) boxes per day for (\d+) days?"),
        ("receive", r"receives? (\d+) deliveries of (\d+) boxes each"),
    )
    for kind, pattern in patterns:
        for hit in re.finditer(pattern, tail, re.I):
            events.append((hit.start(), kind, tuple(map(int, hit.groups()))))
            matched_spans.append(hit.span())
    if not events:
        return None
    uncovered = list(tail)
    for start, end in matched_spans:
        uncovered[start:end] = " " * (end - start)
    if re.search(r"\b(?:split|doubled|ships?|receives?|adds?|subtracts?|multiplied|divided)\b",
                 "".join(uncovered), re.I):
        return None
    for _, kind, args in sorted(events):
        if kind == "split":
            stock //= args[0]
        elif kind == "double":
            stock *= 2
        elif kind == "ship":
            stock -= args[0] * args[1]
        elif kind == "receive":
            stock += args[0] * args[1]
    return str(stock)


def csv_answer(instruction: str, workspace: Path) -> str | None:
    question = re.search(
        r"How many units of product ['\"]([^'\"]+)['\"] were sold in months "
        r"(\d+) through (\d+) inclusive\?", instruction, re.I)
    if not question:
        return None
    files = list(workspace.glob("*.csv"))
    if len(files) != 1:
        return None
    product, low, high = question.groups()
    lo, hi = int(low), int(high)
    with files[0].open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if not {"product", "month", "units"}.issubset(rows.fieldnames or []):
            return None
        total = sum(int(row["units"]) for row in rows
                    if row["product"].casefold() == product.casefold()
                    and lo <= int(row["month"]) <= hi)
    return str(total)


def search_answer(instruction: str, workspace: Path) -> str | None:
    kb = workspace / "kb"
    if not kb.is_dir():
        return None
    born = re.search(r"(?:year was|when was) ([A-Za-z]+ [A-Za-z]+) born", instruction, re.I)
    employer_hq = re.search(r"headquarters of ([A-Za-z]+ [A-Za-z]+)'s employer", instruction, re.I)
    target = born.group(1) if born else employer_hq.group(1) if employer_hq else None
    if not target:
        return None
    person = kb / ("person_" + target.lower().replace(" ", "_") + ".md")
    if not person.is_file():
        return None
    body = person.read_text(encoding="utf-8")
    if born:
        found = re.search(r"^Born:\s*(\d{4})\s*$", body, re.M | re.I)
        return found.group(1) if found else None
    employer = re.search(r"^Employer:\s*(.+?)\s*$", body, re.M | re.I)
    if not employer:
        return None
    company = kb / ("company_" + re.sub(r"[^a-z0-9]+", "_", employer.group(1).lower()).strip("_") + ".md")
    if not company.is_file():
        return None
    found = re.search(r"^Headquarters:\s*(.+?)\s*$", company.read_text(encoding="utf-8"), re.M | re.I)
    return found.group(1) if found else None


def deterministic_answer(task: dict, workspace: Path) -> str | None:
    instruction = task.get("instruction", "")
    kind = task.get("task_type", "")
    try:
        if kind == "math":
            return math_answer(instruction)
        if kind == "data_analysis":
            return csv_answer(instruction, workspace)
        if kind == "search":
            return search_answer(instruction, workspace)
    except (ValueError, OSError, ZeroDivisionError):
        return None
    return None


def has_required_files(task: dict, workspace: Path) -> bool:
    kind = task.get("task_type")
    if kind == "search":
        return (workspace / "kb").is_dir()
    if kind == "data_analysis":
        return any(workspace.glob("*.csv"))
    if kind == "coding":
        return any(workspace.glob("test_*.py")) and any(workspace.glob("*.py"))
    return True


class JiuwenLLM:
    def __init__(self) -> None:
        # Import lazily: deterministic tasks work offline and local smoke tests
        # do not need a proxy key. All actual LLM calls use openJiuwen Model.
        # Its default file logger targets ./logs, which is unwritable when the
        # evaluator starts the image with --read-only and --workdir /app.
        from openjiuwen.core.common.logging.log_config import configure_log_config
        configure_log_config({
            "backend": "default", "level": "ERROR", "output": ["console"],
            "interface_output": ["console"], "performance_output": ["console"],
        })
        from openjiuwen.core.foundation.llm.model import Model
        from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ModelRequestConfig

        base = os.environ["LLM_BASE_URL"].rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.model = Model(
            ModelClientConfig(client_provider="OpenAI", api_base=base,
                              api_key=os.environ["LLM_API_KEY"], timeout=90),
            ModelRequestConfig(model="qwen3.6-35b-a3b-noreason"),
        )
        self.spent = 0.0

    async def ask(self, system: str, user: str, *, max_tokens: int = 400,
                  model: str = "qwen3.6-35b-a3b-noreason") -> str:
        response = await self.model.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model, max_tokens=max_tokens, temperature=0, timeout=90,
        )
        usage = response.usage_metadata
        if usage:
            # Provider output_tokens should include reasoning tokens. If a
            # provider reports them separately, account for the larger value.
            self.spent += max(usage.output_tokens, usage.reasoning_tokens) + usage.input_tokens / 10
        return str(response.content or "").strip()


def task_context(workspace: Path, instruction: str, limit: int = 8500) -> str:
    parts: list[str] = []
    query_terms = set(re.findall(r"[a-z]{4,}", instruction.casefold()))
    # Names and company titles in the instruction are generally more useful
    # than alphabetical order in a large knowledge base.
    paths = [p for p in workspace.rglob("*") if p.is_file()]
    paths.sort(key=lambda p: (-len(query_terms.intersection(re.findall(r"[a-z]{4,}", p.stem.casefold()))), str(p)))
    for path in paths:
        if not path.is_file() or path.name == "answer.txt" or path.suffix not in {".md", ".txt", ".py", ".csv", ".json"}:
            continue
        if len(parts) >= 25:
            break
        content = path.read_text(encoding="utf-8", errors="replace")[:1500]
        parts.append(f"### {path.relative_to(workspace)}\n{content}")
        if sum(map(len, parts)) >= limit:
            break
    return "\n".join(parts)[:limit]


async def model_answer(llm: JiuwenLLM, task: dict, workspace: Path) -> str | None:
    kind = task.get("task_type", "")
    if kind == "data_analysis":
        # The model writes a single SELECT, which is run against an in-memory
        # SQLite copy of the CSV. This avoids shipping hundreds of rows in a prompt.
        import sqlite3
        files = list(workspace.glob("*.csv"))
        if len(files) != 1:
            return None
        with files[0].open(newline="", encoding="utf-8-sig") as stream:
            rows = csv.DictReader(stream)
            fields = rows.fieldnames or []
            values = list(rows)
        if not fields or any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", x) for x in fields):
            return None
        sql = await llm.ask(
            "Write exactly one SQLite SELECT query. Table name is data. Return only SQL, no markdown.",
            f"Columns: {fields}. Numeric-looking values are stored as numeric. Question: {task['instruction']}",
            max_tokens=180,
        )
        sql = re.sub(r"^```(?:sql)?|```$", "", sql.strip(), flags=re.I).strip()
        if not re.match(r"^SELECT\b", sql, re.I) or ";" in sql or len(sql) > 600:
            return None
        con = sqlite3.connect(":memory:")
        try:
            steps = 0
            def limit_query() -> int:
                nonlocal steps
                steps += 1
                return 1 if steps > 500 else 0
            con.set_progress_handler(limit_query, 1000)
            con.execute("CREATE TABLE data (" + ",".join('"' + x + '"' for x in fields) + ")")
            con.executemany("INSERT INTO data VALUES (" + ",".join("?" for _ in fields) + ")",
                            [[int(row[x]) if re.fullmatch(r"-?\d+", row[x] or "") else
                              float(row[x]) if re.fullmatch(r"-?\d+\.\d+", row[x] or "") else row[x]
                              for x in fields] for row in values])
            result = con.execute(sql).fetchall()
        except sqlite3.Error:
            return None
        finally:
            con.close()
        if len(result) != 1 or len(result[0]) != 1 or result[0][0] is None:
            return None
        value = result[0][0]
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)

    response = await llm.ask(
        "Solve the task from the supplied files. Return exactly the final one-line answer, or UNSOLVABLE if evidence is missing. Do not explain.",
        f"Instruction: {task.get('instruction', '')}\n\nFiles:\n{task_context(workspace, task.get('instruction', ''))}",
        max_tokens=220,
    )
    response = response.strip().strip("`").strip()
    if not response or response == "UNSOLVABLE" or "\n" in response:
        return None
    if task.get("check") == "exact_int":
        numeric = re.fullmatch(r"-?\d+", response)
        if not numeric:
            return None
    return response


def run_tests(workspace: Path) -> tuple[bool, str]:
    tests = sorted(workspace.glob("test_*.py"))
    if not tests:
        return False, "No tests"
    try:
        p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-p", "test_*.py"],
                           cwd=workspace, text=True, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=18)
        return p.returncode == 0, (p.stdout or "")[-3000:]
    except subprocess.TimeoutExpired:
        return False, "Tests timed out"


def try_known_repairs(workspace: Path) -> bool:
    """Apply sample-family semantics, retaining the patch only if tests pass."""
    source = workspace / "mylib.py"
    if not source.is_file():
        return False
    original = source.read_text(encoding="utf-8")
    known = {
        "reverse_words": "def reverse_words(s):\n    return ' '.join(reversed(s.split()))",
        "total": "def total(xs):\n    return sum(xs)",
        "maximum": "def maximum(xs):\n    return max(xs)",
        "average": "def average(xs):\n    return sum(xs) / len(xs)",
        "first_n": "def first_n(xs, n):\n    return xs[:n]",
        "count_even": "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)",
    }
    try:
        tree = ast.parse(original)
    except SyntaxError:
        return False
    lines = original.splitlines(keepends=True)
    changes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in known and not node.decorator_list:
            changes.append((node.lineno - 1, node.end_lineno, known[node.name] + "\n"))
    if not changes:
        return False
    for start, end, replacement in reversed(changes):
        lines[start:end] = [replacement]
    source.write_text("".join(lines), encoding="utf-8")
    passing, _ = run_tests(workspace)
    if not passing:
        source.write_text(original, encoding="utf-8")
    return passing


async def solve_coding(llm: JiuwenLLM, task: dict, workspace: Path) -> bool:
    passing, feedback = run_tests(workspace)
    if passing:
        return True
    if try_known_repairs(workspace):
        return True
    source = workspace / "mylib.py"
    tests = sorted(workspace.glob("test_*.py"))
    if not source.exists() or not tests:
        return False
    original = source.read_text(encoding="utf-8")
    test_text = "\n".join(p.read_text(encoding="utf-8")[:4500] for p in tests[:3])
    current = original
    for _ in range(2):
        candidate = await llm.ask(
            "Repair the Python module to pass the tests. Return the complete source file only; no markdown or explanation. Do not modify tests.",
            f"Instruction: {task.get('instruction','')}\n\nCurrent mylib.py:\n{current[:6500]}\n\nTests:\n{test_text}\n\nFailure:\n{feedback}",
            max_tokens=1500, model="qwen3.8-27b-noreason",
        )
        candidate = re.sub(r"^```(?:python)?\s*|\s*```$", "", candidate.strip(), flags=re.I)
        try:
            compile(candidate, str(source), "exec")
        except SyntaxError:
            feedback = "Generated source has a syntax error"
            continue
        source.write_text(candidate + "\n", encoding="utf-8")
        passing, feedback = run_tests(workspace)
        if passing:
            return True
        current = candidate
    source.write_text(original, encoding="utf-8")
    return False


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="test deterministic solvers without API")
    args = parser.parse_args()
    root = args.tasks_dir.resolve()
    entries = []
    for task_file in sorted(root.glob("*/task.json")):
        try:
            task = json.loads(task_file.read_text(encoding="utf-8"))
            workspace = task_file.parent / "workspace"
            if not workspace.is_dir():
                workspace = task_file.parent
            if (workspace / "answer.txt").exists() or (task_file.parent / "answer.txt").exists():
                continue
            entries.append((task_file.parent, workspace, task))
        except (OSError, ValueError):
            continue

    start = time.monotonic()
    remaining = []
    for task_dir, workspace, task in sorted(entries, key=lambda e: -float(e[2].get("reward", 0))):
        answer = deterministic_answer(task, workspace)
        if answer is not None:
            atomic_answer(task_dir, workspace, answer)
            print(f"solved {task_dir.name} locally", flush=True)
        elif has_required_files(task, workspace):
            remaining.append((task_dir, workspace, task))
        else:
            print(f"skipped {task_dir.name}: required files missing", flush=True)

    # Known sample-family repairs are attempted before constructing an LLM
    # client. This lets the complete supplied sample set run offline.
    still_remaining = []
    for task_dir, workspace, task in remaining:
        if task.get("task_type") == "coding" and try_known_repairs(workspace):
            atomic_answer(task_dir, workspace, "DONE")
            print(f"solved {task_dir.name} locally", flush=True)
        else:
            still_remaining.append((task_dir, workspace, task))
    remaining = still_remaining
    if args.offline or not remaining:
        return
    if not os.environ.get("LLM_BASE_URL") or not os.environ.get("LLM_API_KEY"):
        raise SystemExit("LLM_BASE_URL and LLM_API_KEY are required for remaining tasks")
    llm = JiuwenLLM()
    # Simple tasks have the best expected reward per token.  Coding takes
    # longer, but may be worth high reward after cheap tasks are exhausted.
    expected_cost = {"math": 110, "search": 150, "data_analysis": 180, "coding": 1800}
    remaining.sort(key=lambda e: -float(e[2].get("reward", 0)) /
                   expected_cost.get(e[2].get("task_type"), 500))
    for task_dir, workspace, task in remaining:
        if time.monotonic() - start > 3300:
            break
        try:
            if task.get("task_type") == "coding":
                answer = "DONE" if await solve_coding(llm, task, workspace) else None
            else:
                answer = await model_answer(llm, task, workspace)
            if answer is not None:
                atomic_answer(task_dir, workspace, answer)
                print(f"solved {task_dir.name} via openJiuwen", flush=True)
            else:
                print(f"skipped {task_dir.name}", flush=True)
        except Exception as exc:
            print(f"failed {task_dir.name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if re.search(r"\b(?:402|429|quota|budget|insufficient credits)\b", str(exc), re.I):
                break
    print(f"estimated weighted tokens spent: {llm.spent:.1f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
