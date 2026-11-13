"""Fresh eight-case SQL development probe of the final image; four 100k pipelines.

Cases are authored after public100 error analysis. This is a targeted regression
probe, not an independent estimate of final-contest accuracy or a causal ablation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analytical_tools import CSVDatabase
from experiments import policy_experiment as base
from experiments.public_experiment import FamilyExecutor
from experiments.build_production_challenge import dumps, digest

SUITE = ROOT / "experiments/suite_sql_feedback"
RUN_ROOT = ROOT / "experiments/runs/2026-11-13-sql-feedback"
RUN = RUN_ROOT / "test"
TAG = "nsrc-policy-v3-final:20261113"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases():
    return [
        ("reserved_identifiers", "Sum Net Amount for Group blue where Cancelled is 0; do not sum Order.",
         {"ledger.csv": "Order,Group,Net Amount,Cancelled\n101,blue,17,0\n102,blue,99,1\n103,blue,23,0\n104,red,80,0\n"},
         "40", 'SELECT SUM("Net Amount") FROM ledger WHERE "Group"=\'blue\' AND Cancelled=0'),
        ("two_table_join", "Sum the order amount for customers in the North region, matching customer_id across both complete tables.",
         {"customers.csv": "customer_id,region\n1,North\n2,South\n3,North\n",
          "orders.csv": "order_id,customer_id,amount\n71,1,12\n72,2,90\n73,3,8\n74,1,5\n"},
         "25", "SELECT SUM(o.amount) FROM orders o JOIN customers c ON c.customer_id=o.customer_id WHERE c.region='North'"),
        ("weighted_average", "Compute the quantity-weighted average price over rows with cancelled=0. Do not use the unweighted average.",
         {"prices.csv": "price,quantity,cancelled\n10,2,0\n16,3,0\n99,100,1\n"},
         "13.6", "SELECT 1.0*SUM(price*quantity)/SUM(quantity) FROM prices WHERE cancelled=0"),
        ("empty_count", "How many rows have status equal to missing? Count rows; zero matching rows means the answer is zero.",
         {"events.csv": "event_id,status\n1,active\n2,closed\n3,active\n"},
         "0", "SELECT COUNT(*) FROM events WHERE status='missing'"),
        ("median", "Return the median reading, averaging the two middle readings if their count is even. Include every row.",
         {"readings.csv": "reading\n4\n11\n2\n20\n9\n6\n"},
         "7.5", "SELECT AVG(reading) FROM (SELECT reading FROM readings ORDER BY reading LIMIT 2 OFFSET 2)"),
        ("quoted_header", "For row label Z return the value in the column whose literal name is Metric \"A\". There is exactly one Z row.",
         {"measures.csv": 'label,"Metric ""A""",other\nY,3,100\nZ,37,200\n'},
         "37", 'SELECT "Metric ""A""" FROM measures WHERE label=\'Z\''),
        ("missing_date", "Using only this table, return the exact age in completed years of employee R on 2026-01-01. No birth dates or ages are provided; do not infer age from employee_id. If not determined return UNSOLVABLE.",
         {"staff.csv": "employee_id,name,department\n71,R,Lab\n72,S,Ops\n"}, None, None),
        ("nonunique_value", "Return the unique value for key X only if all rows for X agree. The table is complete, all rows equally authoritative, and there is no tie-break rule. If there is no unique value return UNSOLVABLE; do not average or choose a row.",
         {"facts.csv": "key,value\nX,5\nX,7\nY,9\n"}, None, None),
    ]


def build():
    if SUITE.exists():
        raise FileExistsError("Refusing to replace an existing diagnostic suite")
    SUITE.mkdir(parents=True)
    manifest, metadata, oracle = [], {}, {}
    for index, (name, question, files, expected, sql) in enumerate(cases()):
        ident = "sqlprobe_" + digest(name + question)[:16]
        directory = SUITE / "tasks" / ident
        workspace = directory / "workspace"
        workspace.mkdir(parents=True)
        for filename, text in files.items():
            (workspace / filename).write_text(text, encoding="utf-8")
        task = {"task_type": "data_analysis", "reward": 1 + index % 3,
                "instruction": question + " Return only the requested numeric value or UNSOLVABLE.\nCase identifier: " + ident + "."}
        (directory / "task.json").write_text(dumps(task), encoding="utf-8")
        if sql:
            db = CSVDatabase(workspace)
            try:
                answer, error = db.execute(sql)
                if error or not base.numeric_equal(answer, expected):
                    raise AssertionError((name, answer, expected, error))
            finally:
                db.close()
        manifest.append({"id": ident, "domain": "data_analysis", "reward": task["reward"],
            "task_path": "tasks/" + ident, "task_sha256": digest(dumps(task)),
            "files_sha256": {n: digest(t) for n, t in files.items()}})
        metadata[ident] = {"split": "test", "task_group": "sql_feedback:" + name,
                           "domain": "data_analysis", "authored_after_public100": True}
        oracle[ident] = {"answerable": expected is not None, "expected": expected,
                        "reference_sql": sql, "hidden_files": {}, "reference_files": {},
                        "reason": name if expected is None else None}
    base.write_json(SUITE / "manifest.json", {"description": __doc__, "tasks": manifest})
    base.write_json(SUITE / "research_metadata.json", metadata)
    base.write_json(SUITE / "oracle.json", oracle)
    base.write_json(SUITE / "validation.json", {"positive_sql_checks": 6, "negative_certificates": {
        "missing_date": "Same table admits multiple birth dates and hence multiple ages.",
        "nonunique_value": "Two equally authoritative X values, 5 and7, contradict requested uniqueness."},
        "expected_positive_values_independent_arithmetic": [40, 25, 13.6, 0, 7.5, 37], "api_calls": 0})
    print("Built eight fresh SQL cases; six numeric references verified offline.")


def configure():
    base.SUITE, base.RUN_ROOT, base.RUN = SUITE, RUN_ROOT, RUN
    base.PHASE, base.NEW_TAG = "test", TAG
    base.BUDGET = RUN_ROOT / "unused-ledger.json"
    base.ThreadPoolExecutor = FamilyExecutor


def verify():
    protocol = base.read(RUN / "protocol.json")
    if protocol["wrapper_sha256"] != sha(__file__):
        raise RuntimeError("Frozen diagnostic wrapper changed")
    if protocol["family_executor_sha256"] != sha(Path(__file__).with_name("public_experiment.py")):
        raise RuntimeError("Frozen family executor changed")


def prepare():
    base.prepare()
    protocol = base.read(RUN / "protocol.json")
    protocol["arms"] = {a: v for a, v in protocol["arms"].items() if a.startswith("adaptive_")}
    protocol["budget_start"] = {a: 0 for a in protocol["arms"]}
    protocol["max_parallel_pipelines"] = 2
    protocol["label"] = "Fresh authored SQL post-validation development regression probe"
    protocol["notes"] = ["Eight fresh tasks; six answerable, two explicit missing/contradictory-data cases.",
        "One independent100k ledger per model; no per-task allowance; no aggregate experiment ceiling.",
        "Chosen after public100 error analysis; not independent contest validation or a causal ablation.",
        "The guarded artifact predates SQL feedback hardening; frozen predictions are diagnostic only."]
    protocol["wrapper_sha256"] = sha(__file__)
    protocol["family_executor_sha256"] = sha(Path(__file__).with_name("public_experiment.py"))
    base.write_json(RUN / "protocol.json", protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "prepare", "run", "report"])
    args = parser.parse_args()
    if args.action == "build":
        build()
    else:
        configure()
        if args.action == "prepare":
            prepare()
        else:
            verify()
            getattr(base, args.action)()
