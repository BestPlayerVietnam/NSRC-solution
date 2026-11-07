import json
import os
import subprocess
import tempfile
from pathlib import Path


def make_task(root, name, kind, reward, instruction, files):
    folder = root / name
    workspace = folder / "workspace"
    workspace.mkdir(parents=True)
    (folder / "task.json").write_text(json.dumps({
        "task_id": name, "task_type": kind, "reward": reward,
        "instruction": instruction,
    }), encoding="utf-8")
    for relative, content in files.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


with tempfile.TemporaryDirectory(prefix="nsrc-calibration-") as tmp:
    root = Path(tmp) / "tasks"
    root.mkdir()
    make_task(root, "math_a", "math", 5,
              "What is 14 multiplied by 6 plus 5? Write only the final answer to answer.txt.", {})
    make_task(root, "math_b", "math", 1,
              "Compute 125 divided by 5, then add 11. Write only the final answer to answer.txt.", {})
    person = "# Irina Smith\nBorn: 1983\nEmployer: Dune Robotics\nMentor: Alice Smith\n"
    make_task(root, "search_a", "search", 2,
              "Who is Irina Smith's mentor? Write only the final answer to answer.txt.",
              {"kb/person_irina_smith.md": person})
    make_task(root, "search_b", "search", 3,
              "Which company employs Irina Smith? Write only the final answer to answer.txt.",
              {"kb/person_irina_smith.md": person})
    sales = "order_id,region,product,month,units,price\n1,west,widget,1,3,10\n2,east,widget,1,5,30\n3,west,widget,2,4,20\n"
    make_task(root, "data_a", "data_analysis", 4,
              "What is the average price of widget orders? Write only the final answer to answer.txt.",
              {"sales.csv": sales})
    make_task(root, "data_b", "data_analysis", 3,
              "How many units of widget were sold in the west region? Write only the final answer to answer.txt.",
              {"sales.csv": sales})
    make_task(root, "coding_a", "coding", 4,
              "Fix mylib.py so all tests pass. Write DONE to answer.txt.",
              {"mylib.py": "def subtract(a, b):\n    return a + b\n",
               "test_mylib.py": "import unittest\nimport mylib as m\nclass T(unittest.TestCase):\n    def test_a(self): self.assertEqual(m.subtract(8, 3), 5)\n    def test_b(self): self.assertEqual(m.subtract(3, 8), -5)\n"})
    make_task(root, "coding_b", "coding", 2,
              "Fix mylib.py so all tests pass. Write DONE to answer.txt.",
              {"mylib.py": "def sign(x):\n    return 1\n",
               "test_mylib.py": "import unittest\nimport mylib as m\nclass T(unittest.TestCase):\n    def test_negative(self): self.assertEqual(m.sign(-2), -1)\n    def test_zero(self): self.assertEqual(m.sign(0), 0)\n    def test_positive(self): self.assertEqual(m.sign(3), 1)\n"})
    make_task(root, "search_missing", "search", 5,
              "In which year was Ghost Person born?", {})
    (root / "search_missing" / "workspace" / "kb").mkdir()
    make_task(root, "coding_contradict", "coding", 5,
              "Fix mylib.py so all tests pass.",
              {"mylib.py": "def f(x): return x\n",
               "test_mylib.py": "import unittest\nimport mylib as m\nclass T(unittest.TestCase):\n    def test_a(self): self.assertEqual(m.f(1), 1)\n    def test_b(self): self.assertEqual(m.f(1), 2)\n"})
    cmd = ["docker", "run", "--rm", "--read-only", "--tmpfs", "/tmp:rw,size=2g",
           "-e", "LLM_BASE_URL", "-e", "LLM_API_KEY", "--mount",
           f"type=bind,source={root},target=/data/tasks", "nsrc-harness:latest",
           "python", "/app/run_agent.py", "--tasks-dir", "/data/tasks",
           "--budget", "10000", "--trace-file", "/data/tasks/trace.jsonl"]
    run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=300, env=os.environ.copy())
    if run.returncode:
        print(run.stdout[-2000:])
        print(run.stderr[-2000:])
        raise SystemExit(run.returncode)
    expected = {"math_a": "89", "math_b": "36", "search_a": "Alice Smith",
                "search_b": "Dune Robotics", "data_a": "20", "data_b": "7",
                "coding_a": "DONE", "coding_b": "DONE"}
    events = [json.loads(line) for line in (root / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = []
    for event in events:
        name = event["task_id"]
        answer_file = root / name / "answer.txt"
        answer = answer_file.read_text(encoding="utf-8").strip() if answer_file.exists() else None
        summary.append({
            "task": name, "predicted": event.get("predicted_tokens", 0),
            "actual": event.get("actual_tokens", 0), "model_calls": event.get("calls", []),
            "correct": answer == expected[name] if name in expected else answer is None,
            "outcome": event["outcome"],
        })
    print(json.dumps(summary, indent=2))
