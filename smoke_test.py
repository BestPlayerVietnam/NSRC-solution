"""Run the supplied eight tasks in an isolated temporary directory."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SAMPLES = Path(__file__).parent / "TASK" / "task-dir"
with tempfile.TemporaryDirectory(prefix="nsrc-sample-") as tmp:
    destination = Path(tmp).resolve() / "tasks"
    shutil.copytree(SAMPLES, destination)
    run = subprocess.run(
        [sys.executable, "-X", "utf8", str(Path(__file__).parent / "run_agent.py"),
         "--tasks-dir", str(destination), "--offline"],
        text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=120,
    )
    print(run.stdout)
    if run.returncode:
        print(run.stderr, file=sys.stderr)
        raise SystemExit(run.returncode)
    for task_file in sorted(destination.glob("*/task.json")):
        task = json.loads(task_file.read_text(encoding="utf-8"))
        folder = task_file.parent
        workspace = folder / "workspace" if (folder / "workspace").is_dir() else folder
        actual = (workspace / "answer.txt").read_text(encoding="utf-8").strip()
        root_actual = (folder / "answer.txt").read_text(encoding="utf-8").strip()
        assert actual == root_actual, task["task_id"]
        if task["task_type"] == "coding":
            assert actual == "DONE", task["task_id"]
        else:
            assert actual == task["expected"], (task["task_id"], actual, task["expected"])
    print("All eight supplied tasks passed")
