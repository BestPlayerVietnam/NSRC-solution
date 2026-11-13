"""Report public100 forecasts using normalized evaluation rows; never fit."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments import analyze_policy as analyzer
from experiments import public_experiment as public


def report():
    public.configure()
    protocol = public.verify_protocol()
    evaluation = public.base.read(public.RUN / 'evaluation.json')
    if evaluation.get('grading', {}).get('id') != public.GRADING_ID:
        raise RuntimeError('Run public_experiment.py report before predictor reporting')
    # Reuse the existing analysis without changing its source or fitting on test.
    previous_run, previous_suite, previous_read = analyzer.RUN, analyzer.SUITE, analyzer.read
    analyzer.RUN, analyzer.SUITE = public.RUN_ROOT, public.SUITE
    def redirected_read(path):
        if Path(path) == public.RUN_ROOT / 'fit_record.json':
            return previous_read(public.FIT_RECORD)
        return previous_read(path)
    analyzer.read = redirected_read
    try:
        with redirect_stdout(io.StringIO()): analyzer.report()
    finally:
        analyzer.RUN, analyzer.SUITE, analyzer.read = previous_run, previous_suite, previous_read
    path = public.RUN_ROOT / 'final_report.json'
    result = previous_read(path)
    result['grading'] = evaluation['grading']
    result['fit_record_source'] = protocol['fit_record_source']
    result['limitations'] = [
        'Adapted public benchmark panel with fresh source groups, not representative hidden-final evidence.',
        'Public sources may have appeared in model pretraining; no contamination-free claim.',
        'All100 tasks are test only; the predictor was fit exclusively on the earlier train/calibration experiment.',
        'Each model/policy pipeline has an independent100k budget; historical ledgers remain separate.',
        'Search primary correctness is official normalized EM over any accepted span; exact metrics are secondary.',
        'Cost metrics use the same adaptive trajectories and only reconciled, complete observations; skipped tasks are excluded.',
    ]
    analyzer.write(path, result)
    print(json.dumps({'prediction':result['prediction'], 'execution':result['execution']},ensure_ascii=False,indent=2))


if __name__ == '__main__': report()
