"""Build an independent 40-task holdout using the frozen suite builder.

No model calls or reference-code execution. The original suite and builder are
read-only inputs. Run from any directory with Python 3.10+.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path

import build_suite as base

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "experiments" / "suite"
OUT = ROOT / "experiments" / "suite_fresh"
QUOTAS = {"test": (8, 2)}
DOMAINS = ("math", "coding", "search", "data_analysis")


def partition(group):
    """This independent suite is entirely held-out test data."""
    return "test"


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_sha(table):
    obj = {
        "header": [norm(x) for x in table["header"]],
        "types": table["types"],
        "rows": [[norm(x) for x in row] for row in table["rows"]],
    }
    return base.digest(json.dumps(obj, ensure_ascii=False, sort_keys=True))


class FreshBuilder(base.Builder):
    def __init__(self, manifest, oracle):
        super().__init__()
        self.old_manifest = manifest
        self.old_oracle = oracle
        self.old_groups = {r["group"] for r in manifest["tasks"]}
        self.old_ids = {r["id"] for r in manifest["tasks"]}
        self.old_input_shas = {r["input_sha256"] for r in manifest["tasks"]}
        self.pending = []
        self.seen_questions = set()
        self.seen_groups = set()
        self.trap_counts = Counter()
        self.old_instructions = set()
        self.old_contexts = set()
        self.old_questions = set()

        self.gsm_questions = {}
        for line in (base.CACHE / "gsm8k_train.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            self.gsm_questions["gsm8k:" + base.digest(item["question"])] = item["question"]
        self.mbpp_questions = {
            "mbpp:" + str(item["task_id"]): item["prompt"]
            for item in json.loads((base.CACHE / "mbpp_sanitized.json").read_text(encoding="utf-8"))
        }
        squad = json.loads((base.CACHE / "squad_dev_v2.json").read_text(encoding="utf-8"))
        self.squad_questions = {}
        self.squad_no = []
        for article in squad["data"]:
            group = "squad2:" + article["title"].strip().casefold()
            for paragraph in article["paragraphs"]:
                ctx = paragraph["context"]
                for qa in paragraph["qas"]:
                    self.squad_questions[qa["id"]] = qa["question"]
                    if not qa.get("is_impossible", False) or not 150 <= len(ctx) <= 1400:
                        continue
                    instruction = (
                        "Use only kb/source.md as evidence. " + qa["question"]
                        + " Return only a short answer copied from that document, or UNSOLVABLE"
                        + " when that document does not contain enough evidence. Other knowledge is not allowed."
                    )
                    self.squad_no.append({
                        "group": group, "instruction": instruction,
                        "files": {"kb/source.md": ctx}, "answer": None,
                        "extra": {"source_question_id": qa["id"], "scope": "kb/source.md",
                                  "source_is_impossible": True},
                        "witness": {"certificate": "SQuAD 2.0 human-annotated unanswerable",
                                    "source_question_id": qa["id"], "scope": "kb/source.md"},
                    })

        with tarfile.open(base.CACHE / "wikisql_data.tar.bz2") as archive:
            member = next(m for m in archive.getmembers()
                          if m.isfile() and m.name.endswith("/dev.tables.jsonl"))
            self.tables = {item["id"]: item for item in
                           (json.loads(line) for line in archive.extractfile(member))}
        self.old_table_shas = {
            table_sha(self.tables[group.split(":", 1)[1]])
            for group in self.old_groups if group.startswith("wikisql:")
        }
        for record in manifest["tasks"]:
            folder = OLD / record["task_path"]
            task = json.loads((folder / "task.json").read_text(encoding="utf-8"))
            self.old_instructions.add(norm(task["instruction"]))
            extra = oracle[record["id"]]
            question = self.original_question(record["source"], record["group"], extra,
                                              task["instruction"])
            self.old_questions.add(norm(question))
            if record["domain"] == "search":
                self.old_contexts.add(base.digest(norm(
                    (folder / "workspace" / "kb" / "source.md").read_text(encoding="utf-8"))))
        self.squad_no.sort(key=lambda c: base.digest(c["group"] + c["instruction"]))

    def original_question(self, source, group, extra, instruction):
        if source == "gsm8k":
            return self.gsm_questions[group]
        if source == "mbpp":
            return self.mbpp_questions[group]
        if source == "squad2":
            return self.squad_questions[extra["source_question_id"]]
        if source == "wikisql":
            return extra["original_question"]
        return instruction

    def eligible(self, candidate, source):
        group = candidate["group"]
        if group in self.old_groups or norm(candidate["instruction"]) in self.old_instructions:
            return False
        question = norm(self.original_question(source, group, candidate.get("extra") or {},
                                              candidate["instruction"]))
        if question in self.old_questions or question in self.seen_questions:
            return False
        if source == "squad2":
            if base.digest(norm(candidate["files"]["kb/source.md"])) in self.old_contexts:
                return False
        if source == "wikisql":
            if table_sha(self.tables[group.split(":", 1)[1]]) in self.old_table_shas:
                return False
        return True

    def pick(self, candidates, source, domain):
        selected = 0
        for raw in sorted(candidates, key=lambda c: base.digest(c["group"] + c["instruction"])):
            candidate = copy.deepcopy(raw)
            candidate.pop("split", None)
            if not self.eligible(candidate, source) or candidate["group"] in self.seen_groups:
                continue
            self.add(source=source, domain=domain, split="test", **candidate)
            selected += 1
            if selected == QUOTAS["test"][0]:
                break
        assert selected == QUOTAS["test"][0], (source, "not enough fresh source groups", selected)

    def fresh_trap(self, domain, index):
        assert index in (0, 1), (domain, index)
        common = " Use only the supplied facts. Return UNSOLVABLE if no unique valid answer exists."
        if domain == "math":
            if index == 0:
                question = (
                    "A customer buys 4 notebooks for 7 dollars each. The order also has a"
                    " nonnegative delivery charge, whose amount is not specified."
                    " What is the total amount paid in dollars?"
                )
                witness = {"unknown": "delivery_charge", "world1": 2, "world2": 9,
                           "answers": [30, 37], "reference_equation": "total = 4 * 7 + delivery_charge"}
                mechanism = "unknown_delivery"
            else:
                question = (
                    "A pump moves water at a constant rate of 12 liters per minute."
                    " It operates for an unspecified positive whole number of minutes."
                    " How many liters does it move?"
                )
                witness = {"unknown": "duration_minutes", "world1": 3, "world2": 8,
                           "answers": [36, 96], "reference_equation": "liters = 12 * duration_minutes"}
                mechanism = "unknown_duration"
            files = {}
        elif domain == "coding":
            x = 101 + index
            if index == 0:
                checks = ["assert y * y < 0"]
                demand = "its result y must satisfy y * y < 0"
                proof = "Every built-in Python int y has a nonnegative exact integer square."
                mechanism = "negative_integer_square"
            else:
                checks = ["assert y % 3 == 0", "assert y % 3 == 1"]
                demand = "the same result y must satisfy both y % 3 == 0 and y % 3 == 1"
                proof = "For one built-in int, y % 3 is one fixed integer and cannot equal both 0 and 1."
                mechanism = "incompatible_modulo_three"
            question = (
                f"Implement a deterministic function f(x) returning a built-in int. For x={x}, {demand}."
                " All checks apply to the same returned value. Modifying tests, globals, built-ins,"
                " operator behavior, or the test framework is forbidden. Return the complete"
                " module, or UNSOLVABLE if no such function exists."
            )
            files = {
                "mylib.py": "def f(x):\n    return x\n",
                "test_mylib.py": f"from mylib import f\ny = f({x})\nassert type(y) is int\n"
                                + "\n".join(checks) + "\n",
            }
            witness = {"certificate": proof, "checks": ["assert type(y) is int"] + checks}
        else:
            assert domain == "data_analysis"
            if index == 0:
                files = {"sales.csv": "row_id,product,units,unit_price_rub\n1,notebook,6,25\n2,folder,4,18\n"}
                question = (
                    "Using only sales.csv, what is the total mass in kilograms of the six notebooks"
                    " in row_id 1? Their positive per-unit mass is not specified. Prices do not"
                    " determine mass. Return a scalar or UNSOLVABLE if it cannot be determined."
                )
                witness = {"unknown": "mass_kg_per_unit", "world1": 0.2, "world2": 0.5,
                           "answers": [1.2, 3.0], "reference_equation": "mass = 6 * mass_kg_per_unit"}
                mechanism = "unknown_item_mass"
            else:
                files = {"sales.csv": "row_id,product,total_mass_kg,destination\n1,folders,12,Omsk\n2,pens,5,Tomsk\n"}
                question = (
                    "Using only sales.csv, compute the shipping charge in rubles for row_id 1."
                    " The charge is total_mass_kg multiplied by an unspecified positive tariff"
                    " in rubles per kilogram, with no fixed fee. Return a scalar or UNSOLVABLE"
                    " if it cannot be determined."
                )
                witness = {"unknown": "tariff_rub_per_kg", "world1": 7, "world2": 11,
                           "answers": [84, 132], "reference_equation": "charge = 12 * tariff_rub_per_kg"}
                mechanism = "unknown_shipping_tariff"
        return {
            "group": "synthetic:fresh_v1:" + mechanism, "instruction": question + common,
            "files": files, "answer": None, "witness": witness,
            "extra": {"mechanism": mechanism},
        }

    def add(self, source, group, domain, instruction, files, answer, split=None, witness=None, extra=None):
        candidate = {"group": group, "instruction": instruction, "files": files, "answer": answer,
                     "witness": witness, "extra": extra or {}}
        if answer is None and domain in ("math", "coding", "data_analysis"):
            assert source in ("synthetic", "wikisql+synthetic")
            candidate = self.fresh_trap(domain, self.trap_counts[domain])
            self.trap_counts[domain] += 1
            source = "synthetic"
        elif answer is None and source == "squad2":
            # The inherited SQuAD loop counts attempted adds. Replace its candidate
            # with an independently filtered real impossible example before adding.
            candidates = [c for c in self.squad_no if self.eligible(c, source)]
            assert candidates, "No unused real SQuAD unanswerable questions remain"
            candidate = copy.deepcopy(next(
                (c for c in candidates if c["group"] not in self.seen_groups), candidates[0]))
        assert (split or partition(candidate["group"])) == "test"
        assert self.eligible(candidate, source), (source, candidate["group"], "overlap")
        candidate = copy.deepcopy(candidate)
        candidate["files"] = {
            name: content.replace("\r\n", "\n").replace("\r", "\n")
            for name, content in sorted(candidate["files"].items())
        }
        question = self.original_question(source, candidate["group"], candidate["extra"],
                                          candidate["instruction"])
        candidate["extra"]["source_question_sha256"] = base.digest(norm(question))
        self.seen_questions.add(norm(question))
        self.seen_groups.add(candidate["group"])
        if source == "wikisql":
            # The solver stores numeric-looking CSV cells as numbers, including
            # originally textual WikiSQL columns. Match that coercion in oracle SQL.
            candidate["extra"]["gold_params"] = [
                float(value) if isinstance(value, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", value) else value
                for value in candidate["extra"]["gold_params"]
            ]
        self.pending.append({"source": source, "domain": domain, "split": "test", **candidate})

    def materialize(self):
        if OUT.exists():
            assert not (OUT / "manifest.json").exists(), "Refusing to overwrite a completed fresh suite"
            assert not (OUT / "oracle.json").exists(), "Refusing to overwrite a completed fresh suite"
            expected = set()
            for candidate in self.pending:
                ident = candidate["domain"] + "_" + base.digest(
                    candidate["group"] + candidate["instruction"])[:12]
                expected.add("tasks/" + ident + "/task.json")
                expected.update("tasks/" + ident + "/workspace/" + name for name in candidate["files"])
            existing = {p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file()}
            assert existing <= expected, ("Unrecognized files in incomplete fresh suite", existing - expected)
        for candidate in self.pending:
            base.Builder.add(self, **candidate)


def main():
    manifest_path = OLD / "manifest.json"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_oracle = json.loads((OLD / "oracle.json").read_text(encoding="utf-8"))
    protected = {p: file_sha(p) for p in OLD.rglob("*") if p.is_file()}
    protected[Path(base.__file__).resolve()] = file_sha(Path(base.__file__).resolve())
    for name in ("run_experiments.py", "predictor.py"):
        path = ROOT / "experiments" / name
        protected[path] = file_sha(path)
    sources = {
        key: {"file": value[0], "url": value[1], "license": value[2],
              "sha256": file_sha(base.CACHE / value[0])}
        for key, value in base.SOURCES.items()
    }
    for key, source in sources.items():
        assert source["sha256"] == old_manifest["sources"][key]["sha256"], (key, "source changed")

    # Only these in-memory module settings change; build_suite.py is not edited.
    base.OUT, base.QUOTAS, base.partition = OUT, QUOTAS, partition
    builder = FreshBuilder(old_manifest, old_oracle)
    base.math_tasks(builder)
    base.mbpp_tasks(builder)
    base.squad_tasks(builder)
    base.sql_tasks(builder)

    counts = Counter((c["domain"], c["answer"] is not None) for c in builder.pending)
    assert len(builder.pending) == 40
    for domain in DOMAINS:
        assert counts[domain, True] == 8 and counts[domain, False] == 2, (domain, counts)
    assert not builder.old_groups.intersection(c["group"] for c in builder.pending)
    assert not builder.old_questions.intersection(builder.seen_questions)
    builder.materialize()

    records = sorted(builder.records, key=lambda r: r["id"])
    assert not builder.old_ids.intersection(r["id"] for r in records)
    assert not builder.old_input_shas.intersection(r["input_sha256"] for r in records)
    by_id = {r["id"]: r for r in records}
    assert len(by_id) == 40
    for record in records:
        folder = OUT / record["task_path"]
        task = json.loads((folder / "task.json").read_text(encoding="utf-8"))
        files = {p.relative_to(folder / "workspace").as_posix(): p.read_text(encoding="utf-8")
                 for p in sorted((folder / "workspace").rglob("*")) if p.is_file()}
        assert record["input_sha256"] == base.digest(base.dumps(task) + base.dumps(files)), record["id"]
        assert task["task_type"] == record["domain"] and record["split"] == "test"

    audit = {
        "baseline_manifest_sha256": protected[manifest_path],
        "baseline_builder_sha256": protected[Path(base.__file__).resolve()],
        "old_task_count": len(old_manifest["tasks"]), "fresh_task_count": len(records),
        "counts": {f"{domain}/test/{answerable}": counts[domain, answerable]
                   for domain in DOMAINS for answerable in (False, True)},
        "group_overlap": 0, "normalized_source_question_overlap": 0,
        "task_id_overlap": 0, "input_sha256_overlap": 0,
        "squad_context_overlap": 0, "wikisql_table_content_overlap": 0,
        "fresh_search_articles": sorted({r["group"] for r in records if r["domain"] == "search"}),
        "fresh_wikisql_tables": sorted({r["group"] for r in records if r["source"] == "wikisql"}),
        "source_sha256_match_baseline": True, "workspace_input_hashes_verified": True,
    }
    fresh_manifest = {
        "version": "nsrc-near-contest-fresh-v1", "seed": 20261112,
        "model_budget_each": 100000, "model_budget_total": 400000,
        "budget_scope": "development budget per model, shared across experiment rounds; no reset",
        "sources": sources, "tasks": records,
        "notes": [
            "Independent 40-task test suite; no old source groups, normalized questions, or task inputs reused.",
            "Original GSM8K, MBPP, SQuAD and WikiSQL selection restrictions are inherited.",
            "SQuAD impossible labels are scoped only to kb/source.md; no label enters a workspace.",
            "Math/coding/CSV negative cases use new mechanisms with private certificates or two-world witnesses.",
            "Gold labels, source references, witnesses and withheld tests are outside agent workspaces.",
            "No model calls or reference-code execution occur during construction.",
            "Public benchmarks may occur in model pretraining; this is not a contamination-free evaluation.",
            "Fresh file contents normalize line endings before materialization. CSV previews can therefore include more nonempty rows than in the original Windows-generated suite; this input-format variation is not hidden.",
        ],
        "audit": audit,
    }
    (OUT / "manifest.json").write_text(base.dumps(fresh_manifest), encoding="utf-8")
    (OUT / "oracle.json").write_text(base.dumps(builder.oracle), encoding="utf-8")
    for path, expected in protected.items():
        assert file_sha(path) == expected, ("protected file changed", str(path))
    for source in sources.values():
        assert file_sha(base.CACHE / source["file"]) == source["sha256"]
    print(base.dumps({"suite": str(OUT), "audit": audit, "sources": sources,
                      "protected_files_unchanged": len(protected)}))


if __name__ == "__main__":
    main()

