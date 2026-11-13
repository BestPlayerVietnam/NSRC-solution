"""Deployment regression tests; no API calls or frozen runtime writes.

Run with project Python:
  python -B experiments/deployment_hardening_test.py --staged
This applies experiments/deployment_hardening.patch ONLY to a temporary copy.
After integrating the patch, omit --staged to check the real runtime.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STAGED = "--staged" in sys.argv
if STAGED:
    sys.argv.remove("--staged")

def apply_staged_patch(original, patch_text):
    """Apply one exact-context unified diff in memory; no filesystem patch tool."""
    lines, diff = original.splitlines(keepends=True), patch_text.splitlines(keepends=True)
    if diff[:2] != ["--- a/run_agent.py\n", "+++ b/run_agent.py\n"]:
        raise AssertionError("Unexpected staged patch target")
    output, cursor, index = [], 0, 2
    while index < len(diff):
        match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n", diff[index])
        if not match:
            raise AssertionError("Unexpected hunk header")
        start = int(match.group(1)) - 1
        old_count = int(match.group(2) or 1)
        new_count = int(match.group(4) or 1)
        if start < cursor:
            raise AssertionError("Overlapping hunks")
        output.extend(lines[cursor:start])
        cursor = start
        old_seen = new_seen = 0
        index += 1
        while index < len(diff) and not diff[index].startswith("@@"):
            raw = diff[index]
            index += 1
            marker, text = raw[:1], raw[1:]
            if marker in (" ", "-"):
                if cursor >= len(lines) or lines[cursor] != text:
                    raise AssertionError("Frozen runtime differs from staged patch context")
                cursor += 1
                old_seen += 1
            if marker in (" ", "+"):
                output.append(text)
                new_seen += 1
            if marker not in (" ", "-", "+"):
                raise AssertionError("Unsupported patch line")
        if old_seen != old_count or new_seen != new_count:
            raise AssertionError("Hunk length mismatch")
    output.extend(lines[cursor:])
    return "".join(output)

def write_task(parent, ident, kind, instruction, files, reward=1):
    folder = parent / ident
    workspace = folder / "workspace"
    workspace.mkdir(parents=True)
    task = {"task_id": ident, "task_type": kind, "instruction": instruction,
            "reward": reward}
    (folder / "task.json").write_text(json.dumps(task), encoding="utf-8")
    for name, content in files.items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return folder, workspace, task

def good_coding(parent):
    return write_task(
        parent, "coding_good", "coding", "Implement increment(x), returning x + 1.",
        {"mylib.py": "def increment(x):\n    return x + 1\n",
         "test_mylib.py": "from mylib import increment\n\ndef test_increment():\n    assert increment(4) == 5\n"},
    )

class DeploymentHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("pytest") is None:
            raise unittest.SkipTest("Use .tools/jiuwen-venv/Scripts/python.exe (pytest required)")
        cls.original_path = ROOT / "run_agent.py"
        cls.original_bytes = cls.original_path.read_bytes()
        cls.stage = tempfile.TemporaryDirectory(
            prefix="deployment-hardening-", dir=ROOT / ".tools")
        cls.stage_path = Path(cls.stage.name).resolve()
        if not cls.stage_path.is_relative_to((ROOT / ".tools").resolve()):
            raise AssertionError("Temporary copy escaped intended .tools directory")
        if STAGED:
            content = apply_staged_patch(
                cls.original_bytes.decode("utf-8").replace("\r\n", "\n"),
                (ROOT / "experiments/deployment_hardening.patch").read_text(encoding="utf-8"))
            cls.agent_path = cls.stage_path / "run_agent.py"
            cls.agent_path.write_text(content, encoding="utf-8", newline="\n")
        else:
            cls.agent_path = cls.original_path
        spec = importlib.util.spec_from_file_location("_deployment_hardening_agent", cls.agent_path)
        cls.agent = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.agent
        spec.loader.exec_module(cls.agent)

    @classmethod
    def tearDownClass(cls):
        if cls.original_path.read_bytes() != cls.original_bytes:
            raise AssertionError("Frozen run_agent.py changed during staged verification")
        cls.stage.cleanup()

    def invoke_offline(self, task_factory):
        with tempfile.TemporaryDirectory(dir=self.stage_path, prefix="case-") as tmp:
            folder = Path(tmp)
            tasks = folder / "tasks"
            tasks.mkdir()
            bad = task_factory(tasks)
            good = good_coding(tasks)
            trace = folder / "trace.jsonl"
            env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
                   if k in os.environ}
            env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT), HOME=str(folder))
            result = subprocess.run(
                [sys.executable, "-B", str(self.agent_path), "--tasks-dir", str(tasks),
                 "--offline", "--trace-file", str(trace)],
                cwd=folder, env=env, capture_output=True, text=True, encoding="utf-8",
                timeout=60, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((good[0] / "answer.txt").read_text().strip(), "DONE")
            self.assertEqual((good[1] / "answer.txt").read_text().strip(), "DONE")
            self.assertFalse((bad[0] / "answer.txt").exists())
            self.assertFalse((bad[1] / "answer.txt").exists())
            events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            bad_events = [e for e in events if e["task_id"] == bad[0].name]
            self.assertTrue(bad_events)
            self.assertTrue(any(e["outcome"] == "task_error" for e in bad_events))
            self.assertFalse(any(e["outcome"] == "impossible" for e in bad_events))
            return bad_events

    def test_large_coding_workspace_does_not_abort_next_task(self):
        def make_bad(tasks):
            files = {"module_" + str(i) + ".py": "VALUE = 1\n" for i in range(101)}
            files["test_public.py"] = "def test_public():\n    assert 1 == 1\n"
            return write_task(tasks, "coding_oversized", "coding",
                              "Preserve the modules and make public tests pass.", files, reward=100)
        events = self.invoke_offline(make_bad)
        self.assertTrue(any(e["phase"] == "local_coding_check"
                            and e["error_type"] == "ValueError" for e in events))

    def test_valid_large_csv_field_does_not_abort_next_task(self):
        def make_bad(tasks):
            content = "product,month,units\n" + "X" * 150000 + ",1,2\n"
            return write_task(tasks, "data_large_field", "data_analysis",
                "How many units of product 'A' were sold in months 1 through 2 inclusive?",
                {"sales.csv": content}, reward=100)
        events = self.invoke_offline(make_bad)
        self.assertTrue(any(e["phase"] == "local_scan" for e in events))

    def test_feature_sql_error_preserves_other_valid_task(self):
        with tempfile.TemporaryDirectory(dir=self.stage_path, prefix="features-") as tmp:
            root = Path(tmp)
            bad = write_task(root, "data_nul_column", "data_analysis", "Count CSV data rows.",
                             {"sales.csv": "a\x00b\n7\n"})
            good = write_task(root, "math_good", "math", "Compute 2 + 2.", {})
            trace = root / "trace.jsonl"
            prepared, features = self.agent.prepare_task_features([bad, good], trace)
            self.assertEqual([item[0].name for item in prepared], ["math_good"])
            self.assertEqual(set(features), {str(good[0])})
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(events[0]["outcome"], "task_error")
            self.assertEqual(events[0]["phase"], "feature_extraction")
            self.assertEqual(events[0]["error_type"], "ProgrammingError")

    def test_planning_stat_error_preserves_valid_indexes(self):
        with tempfile.TemporaryDirectory(dir=self.stage_path, prefix="plans-") as tmp:
            root = Path(tmp)
            bad = write_task(root, "coding_unreadable", "coding", "Implement f.",
                             {"broken.py": "def f():\n    return 1\n"})
            good = write_task(root, "math_good", "math", "Compute 2 + 2.", {})
            trace = root / "trace.jsonl"
            actual_stat = Path.stat
            def failing_stat(path, *args, **kwargs):
                if path == bad[1] / "broken.py":
                    raise OSError("Simulated transient source-stat failure")
                return actual_stat(path, *args, **kwargs)
            def estimate(_folder, workspace, task):
                return self.agent.TaskEstimator().estimate(task, workspace)
            with patch.object(Path, "stat", failing_stat):
                prepared, plans = self.agent.prepare_task_plans([bad, good], estimate, trace)
            self.assertEqual([item[0].name for item in prepared], ["math_good"])
            self.assertEqual(len(plans), 1)
            self.assertEqual(plans[0][1], 0)
            event = json.loads(trace.read_text().splitlines()[0])
            self.assertEqual(event["outcome"], "task_error")
            self.assertEqual(event["phase"], "planning")

if __name__ == "__main__":
    unittest.main()
