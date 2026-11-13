"""Eight fresh authored cases, confidence on/off, four independent model pairs.

This is a targeted development A/B probe, not independent contest accuracy or a
budget-stress benchmark. Each arm has a separate 100k ledger. No fitting occurs.
Only visible tasks are mounted into solvers; references are read by report only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import as_completed
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analytical_tools import CSVDatabase
from coding_tools import is_test, run_tests
from experiments import policy_experiment as base
from experiments.public_experiment import FamilyExecutor

DEFAULT_SUITE = ROOT / 'experiments/suite_confidence_probe'
DEFAULT_RUN = ROOT / 'experiments/runs/2026-11-13-confidence-probe'
LABEL = 'Fresh authored confidence development A/B; not independent contest validation'
SOURCE_PATHS = (Path(__file__), ROOT/'experiments/policy_experiment.py',
                ROOT/'experiments/public_experiment.py', ROOT/'experiments/run_experiments.py')
INTEGRITY = 'Do not modify tests, builtins, standard libraries, import hooks or the test framework.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Refusing to overwrite ' + str(path))
    base.write_json(path, value)


def put_files(directory, files):
    for name, text in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8', newline='\n')


def cases():
    """References are authored here, never passed to the model transport."""
    mixture = (7 * Fraction(3, 4) - Fraction(5, 8)) / 3
    assert mixture == Fraction(37, 24)  # Independently simplified integer fraction.
    reference_sql = ('SELECT 1.0*SUM(l.unit_price*l.quantity)/SUM(l.quantity) '
                     'FROM lines l JOIN batches b ON b.batch_id=l.batch_id '
                     "WHERE b.region='West' AND b.cancelled=0")
    sql_expected = Fraction(11*2 + 19*5 + 7*3, 2+5+3)
    assert sql_expected == Fraction(69, 5)
    positive_source = {
        'package/__init__.py': '',
        'package/selection.py': 'def best_candidate(rows):\n    return rows[0][0] if rows else None\n',
        'mylib.py': 'from package.selection import best_candidate\n\ndef choose(rows):\n    return best_candidate(rows)\n',
        'test_mylib.py': ('from mylib import choose\n\ndef test_public():\n'
                          '    assert choose([]) is None\n'
                          '    assert choose([("z", 4), ("b", 9), ("a", 9)]) == "a"\n'),
    }
    positive_reference = dict(positive_source)
    positive_reference['package/selection.py'] = (
        'def best_candidate(rows):\n'
        '    return min(rows, key=lambda row: (-row[1], row[0]))[0] if rows else None\n')
    hidden = {'test_hidden_selection.py': (
        'from mylib import choose\n\ndef test_hidden():\n'
        '    rows = [("w", -7), ("y", -2), ("x", -2)]\n'
        '    original = list(rows)\n'
        '    assert choose(rows) == "x"\n'
        '    assert rows == original\n'
        '    assert choose([("only", 0)]) == "only"\n'
        '    assert choose([("a", 1), ("b", 3)]) == "b"\n')}
    return [
        dict(name='mixture_fraction', domain='math',
             instruction='Seven vessels each contain 3/4 litre of juice. Exactly 5/8 litre is spilled from the combined juice. The remainder is shared equally among three jars. How many litres does each jar receive? Return the numeric value without rounding.',
             files={}, expected=str(float(mixture)), reference_fraction=[37, 24]),
        dict(name='missing_initial_tickets', domain='math',
             instruction='A ticket desk starts with an unspecified number of tickets, sells exactly 14, then receives exactly 6 additional tickets. How many tickets are at the desk now? No starting count or capacity is provided. Use only these facts; return UNSOLVABLE if the number is not uniquely determined.',
             files={}, expected=None, certificate='Start 20 gives 12 remaining; start 30 gives 22. Both satisfy every fact.'),
        dict(name='dispatch_lookup', domain='search',
             instruction='In the signed dispatch note for shipment LARCH-61, which destination port is recorded? Return exactly the port name from that note.',
             files={'notes/signed_dispatch.md': 'Signed dispatch note. Shipment: LARCH-61. Destination port: Valparaiso. Seal: Q82.\n',
                    'notes/other_dispatch.md': 'Shipment: PINE-92. Destination port: Recife. This is a different shipment.\n'},
             expected='Valparaiso'),
        dict(name='missing_departure_time', domain='search',
             instruction='Using only the complete supplied dispatch note for shipment MAPLE-73, what exact departure time was recorded? Return UNSOLVABLE when no recorded time is supplied; do not infer a time.',
             files={'dispatch.md': 'Complete dispatch note. Shipment: MAPLE-73. Destination port: Bergen. Departure time: not recorded. No other departure-time record is available.\n'},
             expected=None, certificate='The visible record contains no clock time. 08:00 and 17:00 are both possible departures.'),
        dict(name='weighted_join', domain='data_analysis',
             instruction='For noncancelled batches in region West, compute the quantity-weighted mean unit_price across their line items. Join on batch_id. Include every eligible line, exclude other regions and cancelled batches, and return the numeric mean without rounding.',
             files={'batches.csv': 'batch_id,region,cancelled\n1,West,0\n2,East,0\n3,West,1\n4,West,0\n',
                    'lines.csv': 'batch_id,unit_price,quantity\n1,11,2\n1,19,5\n2,100,40\n3,200,30\n4,7,3\n'},
             expected=str(float(sql_expected)), reference_sql=reference_sql,
             manual_reference='(11*2+19*5+7*3)/(2+5+3)=138/10=13.8'),
        dict(name='missing_purchase_price', domain='data_analysis',
             instruction='Compute the exact total profit on the completed sales in this table. Profit is revenue minus purchase cost. No purchase costs, profit margins or relationships determining them are supplied. Use only this complete table and return UNSOLVABLE if exact profit is undetermined.',
             files={'sales.csv': 'sale_id,revenue,status\n91,35,completed\n92,48,completed\n93,100,cancelled\n'},
             expected=None, certificate='Eligible revenue is 83. Total purchase cost 20 gives profit 63; cost 40 gives profit 43. Both agree with the table.'),
        dict(name='max_candidate_multifile', domain='coding',
             instruction='Repair the Python source modules so mylib.choose(rows) returns the name with greatest numeric score; ties use the lexicographically smallest name. Each row is a (name, score) tuple. Return None for an empty sequence. Do not mutate rows. All functions must be stateless and deterministic. '+INTEGRITY,
             files=positive_source, expected='DONE', reference_files=positive_reference,
             hidden_files=hidden),
        dict(name='incompatible_builtin_float', domain='coding',
             instruction='Implement mylib.make_value() returning an exact built-in float. A single returned value must simultaneously satisfy the supplied assertions r < 0 and r >= 0 under ordinary Python built-in comparison semantics. The tests are authoritative and all assertions must hold. All functions must be stateless and deterministic. '+INTEGRITY,
             files={'mylib.py': 'def make_value():\n    return 0.0\n',
                    'test_mylib.py': 'from mylib import make_value\n\ndef test_float_contract():\n    r = make_value()\n    assert type(r) is float\n    assert r < 0\n    assert r >= 0\n'},
             expected=None, certificate='Every ordered exact float has mutually exclusive <0 and >=0 comparisons; NaN satisfies neither. Exact type excludes custom comparison overrides. The integrity contract excludes mutation of tests or builtins.'),
    ]


def build(suite):
    if suite.exists():
        raise FileExistsError('Refusing to replace suite')
    suite.mkdir(parents=True)
    manifest, oracle, checks = [], {}, []
    for case in cases():
        ident = 'confidence_' + hashlib.sha256(case['name'].encode()).hexdigest()[:16]
        directory = suite/'tasks'/ident
        workspace = directory/'workspace'
        workspace.mkdir(parents=True)
        put_files(workspace, case['files'])
        task = {'task_type': case['domain'], 'reward': 1,
                'instruction': case['instruction'] + '\nCase identifier: ' + ident + '.'}
        write_new(directory/'task.json', task)
        manifest.append({'id': ident, 'name': case['name'], 'domain': case['domain'],
                         'reward': 1, 'task_path': 'tasks/'+ident,
                         'task_sha256': sha(directory/'task.json'),
                         'files_sha256': {name: sha(workspace/name) for name in case['files']}})
        gold = {key: value for key, value in case.items()
                if key not in {'name', 'domain', 'instruction', 'files'}}
        gold['answerable'] = case['expected'] is not None
        oracle[ident] = gold
        if case.get('reference_sql'):
            db = CSVDatabase(workspace)
            try:
                answer, error = db.execute(case['reference_sql'])
            finally:
                db.close()
            if error or not base.numeric_equal(answer, case['expected']):
                raise AssertionError('SQL reference disagrees with independent arithmetic')
        if case.get('reference_files'):
            if run_tests(workspace)[0]:
                raise AssertionError('Coding stub must fail public checks')
            with tempfile.TemporaryDirectory(prefix='confidence-reference-') as temporary:
                reference = Path(temporary)
                put_files(reference, {**case['reference_files'], **case['hidden_files']})
                passed, _ = run_tests(reference)
                if not passed:
                    raise AssertionError('Coding reference failed public/hidden tests')
        if case['domain'] == 'search' and case['expected'] is not None:
            if sum(text.count(case['expected']) for text in case['files'].values()) != 1:
                raise AssertionError('Search reference is not unique in evidence')
        checks.append({'id': ident, 'reference_checked': gold['answerable'],
                       'negative_certificate': gold.get('certificate')})
    write_new(suite/'manifest.json', {'label': LABEL, 'tasks': manifest})
    write_new(suite/'oracle.json', oracle)
    write_new(suite/'validation.json', {'api_calls': 0, 'checks': checks,
                                      'positive_count': 4, 'negative_count': 4})
    print('Built eight cases; four positive references checked; four explicit negative certificates.')


def require_image(image):
    if not image or re.fullmatch(r'sha256:[0-9a-f]{64}', image) is None:
        raise ValueError('--image must be a full immutable sha256 image ID')
    if base.image_id(image) != image:
        raise RuntimeError('Docker image ID mismatch')


def configure(suite, run):
    # In-process configuration only: never rewrite the existing runner on disk.
    base.SUITE, base.RUN, base.RUN_ROOT = suite, run, run
    base.BUDGET = run/'unused-ledger.json'


def source_hashes():
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in SOURCE_PATHS}


def verify_visible(suite, manifest):
    expected_dirs = {row['id'] for row in manifest['tasks']}
    if {p.name for p in (suite/'tasks').iterdir()} != expected_dirs:
        raise RuntimeError('Visible task inventory changed')
    for row in manifest['tasks']:
        folder = suite/row['task_path']
        if sha(folder/'task.json') != row['task_sha256']:
            raise RuntimeError('Visible task changed')
        workspace = folder/'workspace'
        actual = {p.relative_to(workspace).as_posix(): sha(p)
                  for p in workspace.rglob('*') if p.is_file()}
        if actual != row['files_sha256']:
            raise RuntimeError('Visible files changed')


def prepare(suite, run, image):
    require_image(image)
    if run.exists():
        raise FileExistsError('Use a fresh output directory for each experiment')
    manifest = read(suite/'manifest.json')
    verify_visible(suite, manifest)
    arms = {}
    for confidence in ('off', 'heuristic'):
        for alias, model in base.MODEL_ALIASES.items():
            arms[confidence+'_'+alias] = {'initial_model': model, 'image': image,
                                         'confidence_policy': confidence}
    protocol = {'label': LABEL, 'created': time.time(), 'image': image, 'arms': arms,
                'task_ids': [row['id'] for row in manifest['tasks']],
                'source_sha256': source_hashes(),
                'suite_sha256': {name: sha(suite/name) for name in
                                 ('manifest.json', 'oracle.json', 'validation.json')},
                'ceilings': {model: 100000 for model in base.MODELS},
                'pipeline_budget': 100000, 'max_parallel_pipelines': 2,
                'cpus_per_pipeline': 4, 'provider_family_serialized': True,
                'primary': ['correct_points', 'total_cost'],
                'notes': ['Compare each on/off pair lexicographically by points, then lower total cost.',
                          'Negative cost, false abstentions, assessments and repeats are secondary diagnostics.',
                          'No per-task quota; no transfer of budget between arms; each arm has a new ledger.',
                          'Single stochastic realization per arm; small authored development set, no significance claim.',
                          'No guarantee of competition under the 100k budget; not a queue-pressure test.',
                          'Initial prediction is the first trace only; no token-prediction accuracy claim.',
                          'Solver mount contains task.json and visible workspace only; oracle stays on host.']}
    write_new(run/'protocol.json', protocol)
    print(json.dumps({'prepared': True, 'image': image, 'arms': list(arms), 'tasks': len(manifest['tasks'])}))


def verify(suite, run, image):
    require_image(image)
    protocol = read(run/'protocol.json')
    if protocol['image'] != image or protocol['source_sha256'] != source_hashes():
        raise RuntimeError('Frozen image or runner dependency changed')
    if protocol['suite_sha256'] != {name: sha(suite/name) for name in protocol['suite_sha256']}:
        raise RuntimeError('Frozen suite changed')
    verify_visible(suite, read(suite/'manifest.json'))
    return protocol


def run_experiment(suite, run, image):
    protocol = verify(suite, run, image)
    if not os.environ.get('LLM_API_KEY'):
        raise RuntimeError('LLM_API_KEY must be injected into this process; it is never printed')
    configure(suite, run)
    manifest = read(suite/'manifest.json')

    def run_arm(arm, item):
        folder = run/arm
        if (folder/'completed.json').exists():
            if read(folder/'completed.json')['returncode']:
                raise RuntimeError('Previously failed arm needs review')
            return
        if folder.exists():
            raise RuntimeError('Partial arm needs review; refusing to reset its budget or answers')
        tasks = folder/'tasks'
        tasks.mkdir(parents=True)
        for row in manifest['tasks']:
            shutil.copytree(suite/row['task_path'], tasks/row['id'])
        relay = base.Relay(protocol)
        relay.arm, relay.selected_model = arm, item['initial_model']
        relay.ledger = base.Ledger(folder/'ledger.json')
        relay.serve()
        name = 'nsrc-confidence-' + arm.replace('_', '-') + '-' + secrets.token_hex(4)
        command = ['docker', 'run', '--rm', '--name', name, '--read-only',
                   '--tmpfs', '/tmp:rw,size=2g', '--cpus', '4', '--memory', '16g',
                   '--pids-limit', '512', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                   '-e', 'HOME=/tmp', '-e', f'LLM_BASE_URL=http://host.docker.internal:{relay.server.server_port}/v1',
                   '-e', 'LLM_API_KEY='+relay.token,
                   '--mount', f'type=bind,source={tasks.resolve()},target=/data/tasks', image,
                   'python', '/app/run_agent.py', '--tasks-dir', '/data/tasks', '--budget', '100000',
                   '--trace-file', '/data/tasks/trace.jsonl', '--initial-model', item['initial_model'],
                   '--fixed-model', '--retry-policy', 'adaptive', '--confidence-policy', item['confidence_policy']]
        started, returncode = time.monotonic(), 124
        try:
            with (folder/'stdout.txt').open('wb') as out, (folder/'stderr.txt').open('wb') as err:
                returncode = subprocess.run(command, stdout=out, stderr=err, timeout=3500).returncode
        except subprocess.TimeoutExpired:
            pass
        finally:
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
            relay.close()
            write_new(folder/'ledger_end.json', relay.ledger.data)
        used = sum(value['charged_units'] for value in relay.ledger.data['models'].values())/10
        write_new(folder/'completed.json', {'returncode': returncode, 'seconds': time.monotonic()-started,
                                          'image': image, 'model': item['initial_model'],
                                          'charged': used, 'finished': time.time()})
        print(json.dumps({'arm': arm, 'returncode': returncode, 'charged': used}), flush=True)
        if returncode or used > 100000:
            raise RuntimeError('Arm failed or exceeded its independent budget: '+arm)

    with base.portalocker.Lock(str(run/'process.lock'), timeout=0):
        with FamilyExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_arm, arm, item) for arm, item in protocol['arms'].items()]
            for future in as_completed(futures):
                future.result()


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def report(suite, run, image):
    protocol = verify(suite, run, image)
    # Do not grade an incomplete comparison or read oracle labels early.
    if not all((run/arm/'completed.json').is_file() and
               read(run/arm/'completed.json')['returncode'] == 0 for arm in protocol['arms']):
        raise RuntimeError('All eight arms must complete before grading')
    manifest, oracle = read(suite/'manifest.json'), read(suite/'oracle.json')
    events = json_lines(run/'calls.jsonl')
    calls = base.charged_calls(events)
    summary = {'label': LABEL, 'protocol_sha256': sha(run/'protocol.json'), 'arms': {},
               'comparison_complete': not any(e['event'] in {'not_started', 'invalid_attribution'} for e in events),
               'no_prediction_accuracy_claim': True, 'paired_results': {}}
    for arm, item in protocol['arms'].items():
        folder = run/arm
        ledger = read(folder/'ledger_end.json')
        arm_calls = [call for call in calls if call['arm'] == arm]
        charged = sum(value['charged_units'] for value in ledger['models'].values())/10
        if charged > 100000 or not math.isclose(charged, sum(call['cost'] for call in arm_calls), abs_tol=.01):
            raise RuntimeError('Call costs differ from ledger or exceed arm budget')
        if any(value['charged_units'] and model != item['initial_model'] for model, value in ledger['models'].items()):
            raise RuntimeError('Unexpected model switch')
        traces = json_lines(folder/'tasks/trace.jsonl')
        jobs = []
        for row in manifest['tasks']:
            gold = oracle[row['id']]
            if not gold.get('hidden_files'):
                continue
            work = folder/row['task_path']/'workspace'
            files = {p.relative_to(work).as_posix(): p.read_text(encoding='utf-8')
                     for p in work.rglob('*.py') if '__pycache__' not in p.parts}
            original = suite/row['task_path']/'workspace'
            files.update({p.relative_to(original).as_posix(): p.read_text(encoding='utf-8')
                          for p in original.rglob('*.py') if is_test(p.relative_to(original))})
            files.update(gold['hidden_files'])
            jobs.append({'id': row['id'], 'files': files})
        checked = {}
        if jobs:
            command = ['docker', 'run', '--rm', '-i', '--network', 'none', '--read-only',
                       '--tmpfs', '/tmp:rw,size=256m', '--cpus', '2', '--memory', '1g',
                       '--pids-limit', '128', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                       image, 'python', '-B', '-c', base.CHECK_CODE]
            result = subprocess.run(command, input=json.dumps(jobs), text=True, encoding='utf-8',
                                    capture_output=True, timeout=150)
            if result.returncode:
                raise RuntimeError('Isolated hidden validator failed')
            checked = json.loads(result.stdout.strip().splitlines()[-1])
        rows = []
        for row in manifest['tasks']:
            ident, gold = row['id'], oracle[row['id']]
            path = folder/row['task_path']/'answer.txt'
            answer = path.read_text(encoding='utf-8').strip() if path.exists() else None
            visits = [trace for trace in traces if trace.get('task_id') == ident]
            records = [call for call in arm_calls if call.get('task') == ident]
            cost = sum(call['cost'] for call in records)
            traced_cost = sum(visit.get('actual_tokens', 0) for visit in visits)
            known = all(call.get('accounting') == 'provider_reconciled' for call in records)
            if known and not math.isclose(cost, traced_cost, abs_tol=.11):
                raise RuntimeError('All-visit trace cost differs from provider usage: '+ident)
            correct = False
            if gold['answerable'] and answer is not None:
                if row['domain'] == 'coding':
                    correct = answer == 'DONE' and checked.get(ident, {}).get('passed', False)
                elif row['domain'] in {'math', 'data_analysis'}:
                    correct = base.numeric_equal(answer, gold['expected'])
                else:
                    correct = answer == gold['expected']
            rows.append({'id': ident, 'domain': row['domain'], 'answerable': gold['answerable'],
                         'correct': correct, 'answered': answer is not None, 'answer': answer,
                         'cost': round(cost, 1), 'traced_cost_all_visits': round(traced_cost, 1),
                         'usage_known': known, 'calls': len(records), 'visits': len(visits),
                         'initial_forecast': visits[0].get('predicted_tokens') if visits else None,
                         'initial_prediction_details': visits[0].get('prediction_details') if visits else None,
                         'outcome': visits[-1].get('outcome') if visits else 'unattempted',
                         'assessments': sum(len(v.get('self_assessments', [])) for v in visits),
                         'retry_decisions': [d for v in visits for d in v.get('retry_decisions', [])]})
        summary['arms'][arm] = {
            'model': item['initial_model'], 'confidence_policy': item['confidence_policy'],
            'points': sum(row['correct'] for row in rows), 'max_points': 4,
            'total_cost': round(charged, 1), 'negative_cost': round(sum(row['cost'] for row in rows if not row['answerable']), 1),
            'positive_no_answer': sum(row['answerable'] and not row['answered'] for row in rows),
            'negative_answered': sum(not row['answerable'] and row['answered'] for row in rows),
            'calls': len(arm_calls), 'assessments': sum(row['assessments'] for row in rows), 'rows': rows}
    for alias in base.MODEL_ALIASES:
        off, on = summary['arms']['off_'+alias], summary['arms']['heuristic_'+alias]
        off_key, on_key = (off['points'], -off['total_cost']), (on['points'], -on['total_cost'])
        summary['paired_results'][alias] = {
            'delta_points_on_minus_off': on['points']-off['points'],
            'delta_cost_on_minus_off': round(on['total_cost']-off['total_cost'], 1),
            'delta_negative_cost': round(on['negative_cost']-off['negative_cost'], 1),
            'lexicographic_winner': 'heuristic' if on_key > off_key else 'off' if on_key < off_key else 'tie'}
    write_new(run/'evaluation.json', summary)
    print(json.dumps({'comparison_complete': summary['comparison_complete'],
                      'pairs': summary['paired_results']}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'prepare', 'run', 'report'])
    parser.add_argument('--suite', type=Path, default=DEFAULT_SUITE)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--image', help='Full immutable sha256 image ID; required after build')
    args = parser.parse_args()
    if args.action == 'build':
        build(args.suite.resolve())
    else:
        if not args.image:
            parser.error('--image is required for prepare/run/report')
        {'prepare': prepare, 'run': run_experiment, 'report': report}[args.action](
            args.suite.resolve(), args.run.resolve(), args.image)


if __name__ == '__main__':
    main()
