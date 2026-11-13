"""Public100 wrapper around the frozen policy runner; no fit operation exists.

Prepare only after the training artifact and its Docker image are frozen.
Gold is parsed only in report(), never during forecasts or model calls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import string
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments import policy_experiment as base

SUITE = ROOT / 'experiments/suite_public100'
RUN_ROOT = ROOT / 'experiments/runs/2026-11-12-public100'
RUN = RUN_ROOT / 'test'
PUBLIC_REPORT = Path(__file__).with_name('public_report.py')
ANALYZER = Path(__file__).with_name('analyze_policy.py')
FIT_RECORD = ROOT / 'experiments/runs/2026-11-12-policy/fit_record.json'
GRADING_ID = 'public100-squad-normalized-em-v1'
PUBLIC_PARALLEL = 2
NORMALIZATION_SOURCE = 'https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py'


class FamilyExecutor(ThreadPoolExecutor):
    """Do not let two pipelines contend for the same provider model family.

    The lock is acquired before a container is created, never while an LLM
    request is already waiting against its inference timeout.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.family_locks = {model.removesuffix('-noreason'): threading.Lock()
                             for model in base.MODELS}

    def submit(self, fn, arm, item):
        lock = self.family_locks[item['initial_model'].removesuffix('-noreason')]
        def run_serialized():
            with lock:
                return fn(arm, item)
        return super().submit(run_serialized)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure():
    base.ROOT = ROOT
    base.SUITE = SUITE
    base.RUN_ROOT = RUN_ROOT
    base.RUN = RUN
    base.PHASE = 'test'
    base.ThreadPoolExecutor = FamilyExecutor
    # Relay replaces this unused initial object with the individual arm ledger.
    base.BUDGET = RUN_ROOT / 'unused-ledger.json'


def normalize_squad(answer):
    """Official SQuAD order: lowercase, ASCII punctuation, articles, whitespace.

    Deliberately use lower(), not casefold(), and string.punctuation rather
    than Unicode punctuation categories, to preserve the published metric.
    """
    value = ''.join(c for c in answer.lower() if c not in string.punctuation)
    value = re.sub(r'\b(a|an|the)\b', ' ', value, flags=re.UNICODE)
    return ' '.join(value.split())


def search_scores(answer, gold):
    if not gold['answerable'] or answer is None:
        return {'normalized': False, 'exact': False, 'canonical': False}
    accepted = gold.get('accepted_answers', [gold['expected']])
    if not isinstance(accepted, list) or not accepted or not all(isinstance(a, str) for a in accepted):
        raise ValueError('Answerable search item requires a nonempty accepted_answers list')
    if not isinstance(answer, str):
        raise TypeError('Prediction must be a string or None')
    # Official get_raw_scores ignores empty-normalized golds, falling back to ''.
    normalized_gold = [normalize_squad(a) for a in accepted if normalize_squad(a)] or ['']
    return {
        'normalized': normalize_squad(answer) in normalized_gold,
        'exact': any(answer.strip() == a.strip() for a in accepted),
        'canonical': answer.strip() == str(gold['expected']).strip(),
    }


def regrade_search(evaluation, oracle):
    """Return a copy; alter search correctness only, never costs or other domains."""
    result = copy.deepcopy(evaluation)
    for arm in result['arms'].values():
        for row in arm['rows']:
            if row['domain'] == 'search':
                scores = search_scores(row.get('answer'), oracle[row['id']])
                row['canonical_exact_correct'] = scores['canonical']
                row['strict_exact_correct'] = scores['exact']
                row['correct'] = scores['normalized']
                row['grading'] = GRADING_ID
            else:
                row['canonical_exact_correct'] = bool(row['correct'])
                row['strict_exact_correct'] = bool(row['correct'])
        rows = arm['rows']
        arm['solved'] = sum(r['correct'] for r in rows)
        arm['reward'] = sum(r['reward'] for r in rows if r['correct'])
        arm['strict_exact_solved'] = sum(r['strict_exact_correct'] for r in rows)
        arm['strict_exact_reward'] = sum(r['reward'] for r in rows if r['strict_exact_correct'])
        arm['canonical_exact_solved'] = sum(r['canonical_exact_correct'] for r in rows)
        arm['canonical_exact_reward'] = sum(r['reward'] for r in rows if r['canonical_exact_correct'])
    result['grading'] = {
        'id': GRADING_ID, 'search_primary': 'official_squad_normalized_em_any_accepted',
        'search_secondary': 'trimmed_case_sensitive_exact_any_accepted',
        'search_legacy_secondary': 'trimmed_case_sensitive_exact_canonical',
        'normalization_source': NORMALIZATION_SOURCE,
        'numeric_and_code': 'unchanged_frozen_policy_runner',
        'unanswerable_tasks_earn_reward': False,
    }
    return result


def verify_hashes(protocol, paths):
    """Small reusable pre-call guard; reading a hash does not parse oracle labels."""
    for key, path in paths.items():
        if key not in protocol or sha(path) != protocol[key]:
            raise RuntimeError('Frozen public input changed or missing: ' + key)


def frozen_paths(include_forecasts=True):
    paths = {
        'wrapper_sha256': Path(__file__), 'runner_sha256': Path(base.__file__),
        'public_report_sha256': PUBLIC_REPORT, 'analyzer_sha256': ANALYZER,
        'manifest_sha256': SUITE / 'manifest.json', 'oracle_sha256': SUITE / 'oracle.json',
        'metadata_sha256': SUITE / 'research_metadata.json',
        'artifact_sha256': ROOT / 'cost_artifact.json', 'fit_record_sha256': FIT_RECORD,
    }
    if include_forecasts: paths['forecasts_sha256'] = RUN / 'forecasts.json'
    return paths


def verify_protocol():
    protocol = base.read(RUN / 'protocol.json')
    verify_hashes(protocol, frozen_paths())
    if protocol.get('grading_id') != GRADING_ID or protocol.get('max_parallel_pipelines') != PUBLIC_PARALLEL:
        raise RuntimeError('Unexpected public grading/concurrency protocol')
    if protocol.get('pipeline_budget') != 100000 or len(protocol['arms']) != 8:
        raise RuntimeError('Expected eight independent100k pipelines')
    if protocol.get('serialize_provider_family') is not True:
        raise RuntimeError('Expected provider-family serialization before container creation')
    return protocol


def prepare():
    configure()
    if not (ROOT / 'cost_artifact.json').is_file() or not FIT_RECORD.is_file():
        raise RuntimeError('Finish train/calibration fitting before public prepare')
    metadata = base.read(SUITE / 'research_metadata.json')
    if len(metadata) != 100 or any(m['split'] != 'test' for m in metadata.values()):
        raise RuntimeError('Public panel must contain exactly100 heldout tasks')
    artifact = base.read(ROOT / 'cost_artifact.json')
    fitted_groups = set(artifact.get('train_groups', [])) | set(artifact.get('calibration_groups', []))
    if fitted_groups & {m['group'] for m in metadata.values()}:
        raise RuntimeError('Public source groups overlap fitted groups')
    # Fail before creating the protocol if a required frozen helper is missing.
    helper_hashes = {key:sha(path) for key,path in frozen_paths(False).items()}
    with redirect_stdout(io.StringIO()):
        base.prepare()
    protocol = base.read(RUN / 'protocol.json')
    protocol.update(helper_hashes)
    protocol.update(label='public100_fresh_source_groups_heldout', grading_id=GRADING_ID,
        max_parallel_pipelines=PUBLIC_PARALLEL, public_wrapper=Path(__file__).name,
        serialize_provider_family=True,
        fit_record_source=str(FIT_RECORD.relative_to(ROOT)))
    protocol['notes'] = [n for n in protocol['notes'] if not n.startswith('Test families')]
    protocol['notes'] += [
        'All100 public tasks are heldout; do not fit or select policies using their outcomes.',
        'New source groups exclude our previous datasets; public benchmark memorization remains possible.',
        'Search uses official SQuAD normalized EM over every accepted source span; exact metrics are secondary.',
        'At most two simultaneous single-model pipelines, each with its own100k ledger; never transfer leftovers.',
        'Oracle labels are parsed only during post-run report, not for forecasts or API feedback.',
    ]
    base.write_json(RUN / 'protocol.json', protocol)
    verify_protocol()
    print(json.dumps({'tasks':len(protocol['task_ids']), 'arms':list(protocol['arms']),
        'max_parallel_pipelines':PUBLIC_PARALLEL, 'grading_id':GRADING_ID, 'wrapper_sha256':protocol['wrapper_sha256']}))


def run():
    configure()
    verify_protocol()  # Must precede base.run, Docker creation and any paid call.
    base.run()


def report():
    configure()
    verify_protocol()
    with redirect_stdout(io.StringIO()):
        base.report()  # Original accounting, numeric comparator and hidden coding tests.
    canonical = base.read(RUN / 'evaluation.json')
    base.write_json(RUN / 'evaluation_canonical_exact_secondary.json', canonical)
    oracle = base.read(SUITE / 'oracle.json')  # First semantic gold access is post-run.
    evaluation = regrade_search(canonical, oracle)
    base.write_json(RUN / 'evaluation.json', evaluation)
    print(json.dumps({arm:{k:v for k,v in data.items() if k!='rows'}
        for arm,data in evaluation['arms'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'report'])
    parser.add_argument('--phase', choices=['test'], default='test')
    globals()[parser.parse_args().action]()
