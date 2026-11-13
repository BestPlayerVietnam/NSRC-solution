"""Post-fix development check: eight retained SQL cases plus one new ratio case.

The eight retained cases motivated the fix. Results are regression evidence,
not an independently selected accuracy estimate or a causal model comparison.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import sql_feedback_experiment as probe

SUITE = ROOT / "experiments/suite_sql_guards"
RUN_ROOT = ROOT / "experiments/runs/2026-11-13-sql-guards-probe"
TAG = "nsrc-policy-v3-sqlguards:20261113"
original_cases = probe.cases


def extended_cases():
    return original_cases() + [
        ("fresh_hour_weighted_rate", "Compute the average hourly_rate weighted by hours over all rows in billing.csv. Keep the fractional part; do not round to an integer.",
         {"billing.csv": "worker,hours,hourly_rate\nN,2,7\nP,3,16\n"},
         "12.4", "SELECT 1.0*SUM(hours*hourly_rate)/SUM(hours) FROM billing")]


def configure():
    probe.SUITE, probe.RUN_ROOT, probe.RUN = SUITE, RUN_ROOT, RUN_ROOT / "test"
    probe.TAG = TAG
    probe.cases = extended_cases
    probe.configure()


def prepare():
    probe.prepare()
    path = probe.RUN / "protocol.json"
    protocol = probe.base.read(path)
    protocol["label"] = "SQL guards post-fix development probe; eight retained and one new task"
    protocol["followup_sha256"] = probe.sha(__file__)
    protocol["notes"] = [
        "Eight retained development cases and one fresh weighted-rate question with new table/schema/values.",
        "Earlier errors motivated the SQL prompt/result guards; retained-case gains are not independent validation.",
        "Four single-model pipelines, each with an independent100k ledger and no per-task allowance.",
        "Original image, protocols, results, artifacts and ledgers remain unchanged.",
        "Cost artifact predates these hardening fixes; forecasts are diagnostics, not a new calibration claim."]
    probe.base.write_json(path, protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "prepare", "run", "report"])
    args = parser.parse_args()
    configure()
    if args.action == "build":
        probe.build()
        path = SUITE / "validation.json"
        validation = probe.base.read(path)
        validation["positive_sql_checks"] = 7
        validation["expected_positive_values_independent_arithmetic"].append(12.4)
        validation["scope"] = "Eight retained cases; ninth authored after previous errors, with new schema and values."
        probe.base.write_json(path, validation)
        print("Followup contains nine cases: seven verified numeric references and two negatives.")
    elif args.action == "prepare":
        prepare()
    else:
        probe.verify()
        protocol = probe.base.read(probe.RUN / "protocol.json")
        if protocol["followup_sha256"] != probe.sha(__file__):
            raise RuntimeError("Frozen followup wrapper changed")
        getattr(probe.base, args.action)()
