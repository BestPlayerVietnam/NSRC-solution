"""Read-only applicability check on suite_policy TRAIN only; never runs task code."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from collections import Counter
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "experiments/suite_policy"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def selected_oracles(text, wanted):
    """Decode only selected TRAIN values; skip other object values lexically."""
    decoder = json.JSONDecoder()
    index = 0
    def whitespace(i):
        while i < len(text) and text[i].isspace():
            i += 1
        return i
    index = whitespace(index)
    if text[index] != "{":
        raise ValueError("Expected oracle object")
    index += 1
    selected = {}
    while True:
        index = whitespace(index)
        if text[index] == "}":
            break
        key, index = decoder.raw_decode(text, index)
        index = whitespace(index)
        if text[index] != ":":
            raise ValueError("Missing oracle colon")
        index = whitespace(index + 1)
        if key in wanted:
            selected[key], index = decoder.raw_decode(text, index)
        else:
            # Oracle values are objects. Do not deserialize held-out labels,
            # expected answers, hidden tests, or reference implementations.
            if text[index] != "{":
                raise ValueError("Expected an object-valued oracle entry")
            depth, quoted, escaped = 0, False, False
            while index < len(text):
                char = text[index]
                index += 1
                if quoted:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = False
                elif char == '"':
                    quoted = True
                elif char in "[{":
                    depth += 1
                elif char in "]}":
                    depth -= 1
                    if depth == 0:
                        break
        index = whitespace(index)
        if text[index] == ",":
            index += 1
        elif text[index] != "}":
            raise ValueError("Malformed oracle object")
    if set(selected) != set(wanted):
        raise ValueError("Missing selected TRAIN oracle entries")
    return selected

def main():
    metadata_path = SUITE / "research_metadata.json"
    manifest_path = SUITE / "manifest.json"
    oracle_path = SUITE / "oracle.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    train_ids = {key for key, value in metadata.items() if value["split"] == "train"}
    if "policy_62ab3bfa6706510a" not in train_ids:
        raise AssertionError("Authorized ragged fixture is not TRAIN")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    records = [r for r in manifest["tasks"] if r["id"] in train_ids]
    oracle = selected_oracles(oracle_path.read_text(encoding="utf-8-sig"), train_ids)
    spec = importlib.util.spec_from_file_location("_constraints_train_check", ROOT / "result_constraints.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    counts, results = Counter(), []
    for record in records:
        ident = record["id"]
        task_dir = SUITE / record["task_path"]
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8-sig"))
        report = module.analyze_task(task, task_dir / "workspace")
        answerable = bool(oracle[ident]["answerable"])
        counts["train_tasks"] += 1
        counts["coding_tasks"] += task["task_type"] == "coding"
        counts["answerable"] += answerable
        counts["unanswerable"] += not answerable
        counts["coding_unanswerable"] += task["task_type"] == "coding" and not answerable
        counts["hard_certificates"] += report.hard_impossible
        counts["false_hard_skips"] += report.hard_impossible and answerable
        counts["correct_hard_certificates"] += report.hard_impossible and not answerable
        if task["task_type"] == "coding":
            results.append({"id": ident, "answerable": answerable,
                            "analysis": asdict(report)})
    result = {
        "scope": "suite_policy TRAIN only; no CAL/TEST oracle values decoded; no task code executed",
        "api_calls": 0, "summary": dict(counts), "coding_results": results,
        "source_sha256": {
            "result_constraints.py": sha(ROOT / "result_constraints.py"),
            "manifest.json": sha(manifest_path),
            "research_metadata.json": sha(metadata_path),
            "oracle.json": sha(oracle_path),
        },
        "limitations": [
            "Targeted authored TRAIN distribution; not a hidden-final false-positive estimate.",
            "Builtin list lengths remain unknown because future implementations may mutate lists.",
            "No runtime integration; no measured model-token saving is claimed.",
        ],
    }
    output = ROOT / "experiments/result_constraints_train.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps(result["summary"], sort_keys=True))

if __name__ == "__main__":
    main()
